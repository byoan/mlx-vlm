"""Audit saved-state Metal addresses without importing MLX or launching a GPU.

Run directly with Python's standard library. The kernel factory is extracted
from the real source and supplied a recorder instead of a Metal compiler, so
this also detects out-of-bounds regressions safely before numerical tests run.
"""

import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[1] / "models/qwen3_5/gated_delta.py"


def kernel_source(masked):
    module = ast.parse(SOURCE.read_text())
    factory = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_make_gated_delta_with_states_kernel"
    )
    namespace = {
        "mx": SimpleNamespace(
            metal=SimpleNamespace(is_available=lambda: True),
            fast=SimpleNamespace(metal_kernel=lambda **kwargs: kwargs),
        )
    }
    exec(
        compile(ast.Module(body=[factory], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace[factory.name](masked)["source"]


def state_intervals(source, batch, steps, saved_steps, heads, value_dim, key_dim):
    """Return the write interval of each contiguous [Dv, Dk] head snapshot."""
    base = re.search(r"states \+= ([^;]+);", source).group(1)
    stride = re.search(r"states_ \+= ([^;]+);", source).group(1)
    for row in range(batch):
        for step in range(saved_steps):
            for head in range(heads):
                variables = {
                    "b_idx": row,
                    "T": steps,
                    "StateT": saved_steps,
                    "Hv": heads,
                    "hv_idx": head,
                    "Dv": value_dim,
                    "Dk": key_dim,
                }
                start = eval(base, {"__builtins__": {}}, variables)
                start += step * eval(stride, {"__builtins__": {}}, variables)
                yield row, step, head, start, start + value_dim * key_dim


class SavedStateLayoutTests(unittest.TestCase):
    def assert_layout(
        self, source, batch, steps, saved_steps, heads, value_dim, key_dim
    ):
        total = batch * saved_steps * heads * value_dim * key_dim
        intervals = []
        for row, step, head, start, end in state_intervals(
            source, batch, steps, saved_steps, heads, value_dim, key_dim
        ):
            expected = ((row * saved_steps + step) * heads + head) * value_dim * key_dim
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(
                end, total, f"row {row}, step {step}: write exceeds allocation"
            )
            self.assertEqual(
                start, expected, f"row {row}, step {step}: wrong saved-state stride"
            )
            intervals.append((start, end))
        # Every allocated element must be written once, with no holes/overlaps.
        cursor = 0
        for start, end in sorted(intervals):
            self.assertEqual(start, cursor)
            cursor = end
        self.assertEqual(cursor, total)

    def test_full_and_shortened_state_layouts(self):
        for masked in (False, True):
            source = kernel_source(masked)
            for batch in (1, 2, 3, 4, 5, 6, 7, 8, 16, 32):
                for steps in (1, 2, 3, 8):
                    for saved_steps in range(steps + 1):
                        with self.subTest(
                            masked=masked,
                            batch=batch,
                            steps=steps,
                            saved_steps=saved_steps,
                        ):
                            self.assert_layout(
                                source, batch, steps, saved_steps, 2, 4, 32
                            )

    def test_six_request_qwen35b_mtp_layout(self):
        # Two draft proposals: verify three tokens but retain only two states.
        # [6, 2, 32, 128, 128] float32 = 24 MiB of intermediate-state storage.
        for masked in (False, True):
            with self.subTest(masked=masked):
                self.assert_layout(kernel_source(masked), 6, 3, 2, 32, 128, 128)


if __name__ == "__main__":
    unittest.main()
