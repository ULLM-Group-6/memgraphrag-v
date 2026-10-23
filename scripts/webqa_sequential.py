#!/usr/bin/env python3
"""Conditional embedding attempts, dedicated assembly, then tuning and test."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import time

CODE = ("submit_webqa_sequential.sh", "webqa_sequential.sbatch",
        "webqa_sequential.py", "webqa_embed_checkpointed.py", "webqa_fast_stage.py",
        "webqa_fast_validation.py", "webqa_validation_cache.py", "webqa_fast_qa.py")
YIELD = 75


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), message, flush=True)


def signature(run_id):
    return {"version": 1, "run_id": run_id,
            "artifacts": str(Path(os.environ["MGRV_ARTIFACTS"]).resolve()),
            "config": str(Path(os.environ["MGRV_RUN_CONFIG"]).resolve()),
            "config_sha256": sha(os.environ["MGRV_RUN_CONFIG"]),
            "code": {name: sha(Path(__file__).parent / name) for name in CODE}}


def check_previous(previous, root=None, registration=None):
    """Only time exhaustion can start the backup embedding attempt."""
    if previous == "-":
        return
    if not previous.isdigit():
        raise ValueError("Invalid predecessor job ID")
    for attempt in range(6):
        result = subprocess.run(["sacct", "-X", "-n", "-P", "-j", previous,
                                 "--format=JobIDRaw,User,State,ExitCode"],
                                check=True, capture_output=True, text=True)
        for line in result.stdout.splitlines():
            fields = line.strip().split("|")
            if len(fields) < 4 or fields[0] != previous:
                continue
            _, user, state, code = fields[:4]
            state = state.split()[0].rstrip("+")
            if user != os.environ["USER"]:
                raise RuntimeError("Predecessor belongs to a different user")
            yielded = False
            if state == "FAILED" and code == "75:0" and root is not None:
                marker = Path(root) / "index/parallel-v2/sequential-v1/attempts" / f"{previous}.json"
                if marker.exists():
                    value = json.loads(marker.read_text())
                    yielded = (value.get("state") == "INCOMPLETE_TIME_BUDGET"
                               and value.get("job_id") == previous
                               and value.get("attempt") == 1
                               and value.get("registration_sha256") == sha(registration))
            if state == "TIMEOUT" or yielded:
                log(f"Previous embedding attempt {previous}: {state} ({code}); continuing saved progress")
                return
            if state not in ("RUNNING", "COMPLETING", "PENDING"):
                raise RuntimeError(f"Previous embedding job {previous} ended {state} ({code}); backup refused. Inspect its log")
        # Accounting can briefly lag behind dependency release. No unsafe fallback.
        if attempt < 5:
            time.sleep(5)
    raise RuntimeError(f"No eligible timeout/checkpoint accounting record for {previous}; inspect before retrying")


@contextmanager
def chain_lock(root):
    import fcntl
    path = root / "index/parallel-v2/sequential-v1/run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another sequential job is still writing this index") from None
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


class Coordinator:
    def __init__(self, args):
        self.args = args
        self.root = Path(os.environ["MGRV_ARTIFACTS"]).resolve()
        self.deadline = float(os.environ["MGRV_SEQ_DEADLINE"])
        self.logs = Path("logs") / f"webqa-seq-{os.environ['SLURM_JOB_ID']}"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.number = 0

    def status(self, state, message):
        """Human-readable log plus a marker proving an intentional checkpoint exit."""
        log(message)
        directory = self.root / "index/parallel-v2/sequential-v1/attempts"
        directory.mkdir(parents=True, exist_ok=True)
        job = os.environ["SLURM_JOB_ID"]
        path = directory / f"{job}.json"
        value = {"job_id": job, "mode": self.args.mode, "attempt": self.args.attempt,
                 "state": state, "message": message, "run_id": self.args.run_id,
                 "registration_sha256": sha(self.args.registration), "time": time.time()}
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as file:
            json.dump(value, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)

    def timeout_warning(self, *_):
        if self.args.mode == "embed":
            suffix = ("Backup attempt may resume saved checkpoints." if self.args.attempt == 1 else
                      "If this job times out, BOTH embedding windows are exhausted; assembly, tuning and test MUST NOT START.")
            # Logging only: a batch finishing in the final two minutes may still succeed.
            log("TIME LIMIT APPROACHING; embedding completion not yet confirmed. " + suffix)

    def interrupted(self, signum, _frame):
        if self.args.mode == "embed":
            message = "EMBEDDING INTERRUPTED: this job did not exit successfully; assembly remains blocked. "
            if self.args.attempt == 2:
                message += "Both embedding attempts ended without success. Inspect sacct for TIMEOUT versus failure/cancellation; resume from saved checkpoints."
            # Avoid buffered I/O reentrancy in a signal handler.
            os.write(2, (time.strftime("%Y-%m-%dT%H:%M:%S%z ") + message + "\n").encode())
        raise SystemExit(128 + signum)

    def step(self, action, worker=0, candidate=0):
        self.number += 1
        if action == "embed-checkpointed":
            command = [os.environ["MGRV_VISION_PYTHON"], "scripts/webqa_embed_checkpointed.py",
                       "--config", os.environ["MGRV_RUN_CONFIG"], "--artifacts", str(self.root),
                       "--deadline", str(self.deadline), "--run-id", self.args.run_id]
        else:
            command = [os.environ["MGRV_TEXT_PYTHON"], "scripts/webqa_fast_stage.py", action,
                       "--config", os.environ["MGRV_RUN_CONFIG"], "--artifacts", str(self.root),
                       "--run-id", self.args.run_id, "--split", "dev",
                       "--worker", str(worker), "--candidate", str(candidate)]
        output = self.logs / f"{self.number:02d}-{action}-w{worker}-c{candidate}.log"
        log(f"START {action}, worker={worker}, candidate={candidate}; log={output}")
        result = subprocess.run(["srun", "--exact", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=9", "--gpus=1", "--mem=96G", "--kill-on-bad-exit=1",
            f"--output={output}", f"--error={output}", *command])
        if action == "embed-checkpointed" and result.returncode == YIELD:
            return False
        if result.returncode:
            raise RuntimeError(f"{action} failed ({result.returncode}); inspect {output}")
        log(f"DONE {action}")
        return True

    def embed(self):
        stages = json.loads((self.root / "index/manifest.json").read_text())["stages"]
        if "ground" not in stages:
            raise RuntimeError("This continuation requires your already completed ground stage")
        if not self.step("embed-checkpointed"):
            suffix = ("Backup embedding attempt will resume; assembly stays blocked." if self.args.attempt == 1 else
                      "BOTH embedding windows exhausted with work remaining. Assembly, tuning and test MUST NOT START. Resume embedding with the same artifacts/run ID.")
            self.status("INCOMPLETE_TIME_BUDGET", "Embedding stopped at a saved checkpoint before the time limit. " + suffix)
            return YIELD
        self.status("EMBED_COMPLETE", "EMBEDDING COMPLETE: exiting successfully now; release assembly and skip any backup attempt.")
        return 0

    def assemble(self):
        # Complete worker checkpoints are required and verified by the finalizer.
        # No encoding is performed in this six-hour allocation.
        self.step("embed-finalize")
        self.step("assemble")
        self.status("INDEX_COMPLETE", "INDEX COMPLETE: finalization and graph assembly succeeded; tuning may start.")
        return 0

    def run(self):
        saved = json.loads(Path(self.args.registration).read_text())
        if saved != signature(self.args.run_id):
            raise RuntimeError("Sequential scripts/config/paths changed after submission; inspect before running")
        if self.args.mode == "embed" and self.args.attempt == 2:
            check_previous(self.args.previous, self.root, self.args.registration)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, self.timeout_warning)
        signal.signal(signal.SIGTERM, self.interrupted)
        with chain_lock(self.root):
            self.step("prepare-auto")
            if self.args.mode == "embed":
                return self.embed()
            if self.args.mode == "assemble":
                return self.assemble()
            stages = json.loads((self.root / "index/manifest.json").read_text())["stages"]
            if not {"text", "ground", "embed", "assemble"}.issubset(stages):
                raise RuntimeError("Index is incomplete; tuning/test cannot start")
            if self.args.mode == "tune":
                for candidate in range(3):
                    for worker in range(2):
                        self.step("dev-retrieve", worker, candidate)
                    self.step("dev-collect", candidate=candidate)
                self.step("select-beta")
            else:
                for worker in range(2):
                    self.step("test-qa", worker)
                self.step("test-collect")
            log(f"COMPLETED {self.args.mode}")
            return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--mode", choices=("embed", "assemble", "tune", "test"))
    parser.add_argument("--previous", default="-")
    parser.add_argument("--attempt", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if args.command == "register":
        path = Path(args.registration)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as file:
            json.dump(signature(args.run_id), file, indent=2)
        print(f"Recorded scripts/config: {path}")
    else:
        if args.mode is None:
            parser.error("run requires --mode")
        if args.mode == "embed" and (args.attempt == 0 or (args.attempt == 2 and args.previous == "-")):
            parser.error("embed needs attempt 1 or 2; attempt 2 also requires --previous")
        return Coordinator(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
