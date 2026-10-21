#!/usr/bin/env python3
"""Five allocations, one GPU at a time; model work stays in existing environments."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

CODE = ("submit_webqa_sequential.sh", "webqa_sequential.sbatch",
        "webqa_sequential.py", "webqa_embed_checkpointed.py")
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


def check_previous(previous):
    """afterany permits continuation after a timeout, but never hides real failures."""
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
            if state == "TIMEOUT" or (state == "COMPLETED" and code == "0:0"):
                log(f"Previous indexing slice {previous}: {state}; continuing saved progress")
                return
            if state not in ("RUNNING", "COMPLETING", "PENDING"):
                raise RuntimeError(f"Previous indexing job {previous} ended {state} ({code}); inspect its log")
        # Accounting can briefly lag behind dependency release. No unsafe fallback.
        if attempt < 5:
            time.sleep(5)
    raise RuntimeError(f"No completed/timeout accounting record for {previous}; inspect before retrying")


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

    def step(self, action, worker=0, candidate=0):
        self.number += 1
        if action == "embed-checkpointed":
            command = [os.environ["MGRV_VISION_PYTHON"], "scripts/webqa_embed_checkpointed.py",
                       "--config", os.environ["MGRV_RUN_CONFIG"], "--artifacts", str(self.root),
                       "--deadline", str(self.deadline)]
        else:
            command = [os.environ["MGRV_TEXT_PYTHON"], "scripts/webqa_parallel.py", action,
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

    def yield_index(self):
        if self.args.last_index == 1:
            raise RuntimeError("Index still incomplete after the last slice; tuning is blocked. "
                               "Resume indexing with the same artifacts and run ID.")
        log("Index slice finished at a checkpoint; next queued indexing job will continue")

    def index(self):
        stages = json.loads((self.root / "index/manifest.json").read_text())["stages"]
        if "ground" not in stages:
            raise RuntimeError("This continuation requires your already completed ground stage")
        if "assemble" in stages:
            # Existing implementation verifies cached outputs and dependencies.
            self.step("assemble")
            log("Index already complete and validated; this slice has no new indexing work")
            return
        if "embed" not in stages and not self.step("embed-checkpointed"):
            self.yield_index()
            return
        for action in ("embed-finalize", "assemble"):
            # Start canonical writes only with >=40 min to the HARD 3h limit.
            # This is a margin, not a guarantee of their duration.
            if time.time() + 1800 >= self.deadline:
                self.yield_index()
                return
            self.step(action)
        log("INDEX COMPLETE; all subsequent index slices will only validate/reuse it")

    def run(self):
        saved = json.loads(Path(self.args.registration).read_text())
        if saved != signature(self.args.run_id):
            raise RuntimeError("Sequential scripts/config/paths changed after submission; inspect before running")
        check_previous(self.args.previous)
        with chain_lock(self.root):
            self.step("prepare-auto")
            if self.args.mode == "index":
                self.index()
                return
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--mode", choices=("index", "tune", "test"))
    parser.add_argument("--previous", default="-")
    parser.add_argument("--last-index", type=int, choices=(0, 1), default=0)
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
        Coordinator(args).run()


if __name__ == "__main__":
    main()
