from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc"


def test_marlin_type_ids_header_contains_expected_aliases():
    header = (CSRC / "marlin_type_ids.h").read_text()
    assert "using vllm::FP16_ID;" in header
    assert "using vllm::BF16_ID;" in header
    assert "using vllm::U4B8_ID;" in header
    assert "using vllm::U8B128_ID;" in header
    assert "using vllm::U4_ID;" in header
    assert "using vllm::U8_ID;" in header


def test_marlin_cuda_files_use_shared_type_ids():
    dense = (CSRC / "marlin_dense.cu").read_text()
    moe = (CSRC / "marlin_moe_wna16.cu").read_text()

    assert '#include "marlin_type_ids.h"' in dense
    assert '#include "marlin_type_ids.h"' in moe

    for local_alias in (
        "using vllm::FP16_ID;",
        "using vllm::BF16_ID;",
        "using vllm::U4B8_ID;",
        "using vllm::U8B128_ID;",
    ):
        assert local_alias not in dense
        assert local_alias not in moe

    # Dense path also used to duplicate U4/U8 aliases locally.
    assert "using vllm::U4_ID;" not in dense
    assert "using vllm::U8_ID;" not in dense


def test_marlin_dispatch_shape_list_is_shared_between_dense_and_moe():
    shapes = (CSRC / "marlin_kernel_shapes.h").read_text()
    dense = (CSRC / "marlin_dense.cu").read_text()
    moe = (CSRC / "marlin_moe_wna16.cu").read_text()

    assert "MARLIN_FOR_EACH_SHAPE(OP)" in shapes
    assert "MARLIN_FOR_EACH_SHAPE_WITH_GB(OP, GB)" in shapes

    assert '#include "marlin_kernel_shapes.h"' in dense
    assert '#include "marlin_kernel_shapes.h"' in moe

    assert "MARLIN_FOR_EACH_SHAPE_WITH_GB(DISPATCH_U4B8_FP16, GB)" in dense
    assert "MARLIN_FOR_EACH_SHAPE(DISPATCH_U4B8_FP16)" in moe
