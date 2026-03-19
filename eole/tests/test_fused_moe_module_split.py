import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRITON_INT4_PATH = REPO_ROOT / "eole" / "triton" / "fused_moe_int4.py"
MARLIN_MODULE_PATH = REPO_ROOT / "eole" / "modules" / "fused_moe_marlin.py"
MOE_MODULE_PATH = REPO_ROOT / "eole" / "modules" / "moe.py"


class TestFusedMoeModuleSplit(unittest.TestCase):
    def test_marlin_impl_is_not_in_triton_int4_file(self):
        src = TRITON_INT4_PATH.read_text(encoding="utf-8")
        self.assertNotIn("def _marlin_gemm(", src)
        self.assertNotIn("def fused_experts_marlin_impl(", src)

    def test_marlin_impl_is_in_modules_file_and_moe_uses_it(self):
        marlin_src = MARLIN_MODULE_PATH.read_text(encoding="utf-8")
        moe_src = MOE_MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("def _marlin_gemm(", marlin_src)
        self.assertIn("def fused_experts_marlin_impl(", marlin_src)
        self.assertIn("from eole.modules.fused_moe_marlin import fused_experts_marlin_impl", moe_src)


if __name__ == "__main__":
    unittest.main()
