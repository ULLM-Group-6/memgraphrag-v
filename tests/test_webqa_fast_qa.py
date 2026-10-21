#!/usr/bin/env python3
"""Routing/contract regression checks; no inference or real scheduler calls."""
import ast
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import webqa_run as runner
import webqa_fast_qa as fast


class QA(unittest.TestCase):
    def test_retrieval_loop_matches_original_except_index_loader(self):
        def function(path):
            return next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                        if isinstance(node, ast.FunctionDef) and node.name == "retrieve")
        original = function(Path(runner.__file__))
        updated = function(Path(fast.__file__))
        substitutions = 0
        for node in updated.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "load_index_cached":
                node.value = ast.parse("pipeline.load_index(artifacts, config)", mode="eval").body
                substitutions += 1
        self.assertEqual(substitutions, 1)
        self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(updated, include_attributes=False))

    def worker(self, action):
        config = NS(retrieval=NS(beta=.25))
        artifacts = NS(root=Path("artifacts"), run_dir=lambda name: Path("runs") / name)
        run = NS(config=config, artifacts=artifacts, plan={}, run_id="base", cache=NS(flush=Mock()))
        args = NS(action=action, config="base.yaml", candidate=1, worker=1)
        env = {"MGRV_TEXT_PYTHON": "text", "MGRV_VISION_PYTHON": "vision",
               "MGRV_READER_PYTHON": "reader", "MGRV_READER_MODEL": "reader-model"}
        with patch.dict("os.environ", env), patch.object(fast.api.tuning, "verify_session"), \
             patch.object(fast.api.tuning, "selected_test_config", return_value=(Path("selected.yaml"), config)), \
             patch.object(fast.subprocess, "run") as call, patch.object(runner, "atomic_json"), \
             patch.object(runner, "read_json", return_value={"questions": [["q", "question"]]}), \
             patch.object(fast.api, "required_hash", return_value="hash"):
            fast.run_worker(run, args)
            return [c.args[0] for c in call.call_args_list]

    def test_dev_routes_queries_then_cached_retrieval_without_reader(self):
        commands = self.worker("dev-retrieve")
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][0], "vision")
        self.assertIn("queries", commands[0])
        self.assertIn("webqa_fast_qa.py", Path(commands[1][1]).name)
        self.assertIn("base-dev-beta-050-part-1", commands[1])
        self.assertIn("--base-config", commands[1])
        self.assertEqual(commands[1][commands[1].index("--shard-index") + 1], "1")

    def test_test_routes_selected_beta_and_reader_last(self):
        commands = self.worker("test-qa")
        self.assertEqual(len(commands), 3)
        for command in commands:
            self.assertIn("selected.yaml", command)
            self.assertIn("base-test-part-1", command)
            self.assertEqual(command[command.index("--beta") + 1], "0.25")
        self.assertEqual(commands[2][0], "reader")
        self.assertIn("read", commands[2])
        self.assertIn("reader-model", commands[2])

    def test_test_loader_rejects_beta_not_selected_on_dev(self):
        config = NS(model_dump=lambda **kwargs: {"beta": .25}, retrieval=NS(beta=.25))
        args = NS(base_config="base.yaml", artifacts="artifacts", base_run_id="base", split="test", beta=.75)
        artifacts = NS(index_dir=Path("artifacts/index"))
        run = NS(plan={})
        with patch.object(runner, "context", return_value=(config, artifacts)), \
             patch.object(fast, "HashCache", return_value=nullcontext(NS())), \
             patch.object(fast, "ValidatedRun", return_value=NS(prepare=lambda: run)), \
             patch.object(fast.api.tuning, "selected_test_config", return_value=(Path("selected.yaml"), config)):
            with self.assertRaisesRegex(ValueError, "dev-selected"):
                fast.load_index_cached(args, config, artifacts)


if __name__ == "__main__":
    unittest.main()
