from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc"


def test_marlin_type_ids_header_is_single_alias_source():
    header = (CSRC / "marlin_type_ids.h").read_text()
    assert "using vllm::FP16_ID;" in header
    assert "using vllm::BF16_ID;" in header
    assert "using vllm::U4B8_ID;" in header
    assert "using vllm::U8B128_ID;" in header
    assert "using vllm::U4_ID;" in header
    assert "using vllm::U8_ID;" in header


def test_marlin_cuda_files_include_type_ids_header_without_local_alias_duplication():
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
