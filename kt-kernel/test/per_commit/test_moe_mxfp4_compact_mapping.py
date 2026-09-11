import pytest
import torch


def _has_cpu_flag(flag: str) -> bool:
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            return flag in cpuinfo.read().split()
    except OSError:
        return False


def _source_expert(nibble: int, hidden_size: int, intermediate_size: int):
    packed = (nibble << 4) | nibble
    gate = torch.full(
        (intermediate_size, hidden_size // 2), packed, dtype=torch.uint8
    )
    up = gate.clone()
    down = torch.full(
        (hidden_size, intermediate_size // 2), packed, dtype=torch.uint8
    )
    gate_scale = torch.ones(
        (intermediate_size, hidden_size // 32), dtype=torch.bfloat16
    )
    up_scale = gate_scale.clone()
    down_scale = torch.ones(
        (hidden_size, intermediate_size // 32), dtype=torch.bfloat16
    )
    return gate, up, down, gate_scale, up_scale, down_scale


def _build_moe(moe_module, cpu_infer, sources, logical_ids):
    expert_num = len(logical_ids)
    hidden_size = sources[0][0].shape[1] * 2
    intermediate_size = sources[0][0].shape[0]
    config = moe_module.MOEConfig(
        expert_num, 1, hidden_size, intermediate_size, 0
    )
    config.pool = cpu_infer.backend_
    config.max_len = 2
    config.quant_config.bits = 4
    config.quant_config.group_size = 32
    config.quant_config.zero_point = False
    config.gate_projs = [[source[0].data_ptr() for source in sources]]
    config.up_projs = [[source[1].data_ptr() for source in sources]]
    config.down_projs = [[source[2].data_ptr() for source in sources]]
    config.gate_scales = [[source[3].data_ptr() for source in sources]]
    config.up_scales = [[source[4].data_ptr() for source in sources]]
    config.down_scales = [[source[5].data_ptr() for source in sources]]

    physical_to_logical = torch.tensor(logical_ids, dtype=torch.int64)
    moe = moe_module.AMXFP4_KGroup_MOE(config)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
    cpu_infer.sync()
    return moe, physical_to_logical


def _forward(cpu_infer, moe, hidden_size: int) -> torch.Tensor:
    qlen = torch.tensor([2])
    expert_ids = torch.tensor([[0], [1]], dtype=torch.int64)
    expert_weights = torch.ones((2, 1), dtype=torch.float32)
    hidden_states = torch.full((2, hidden_size), 0.001, dtype=torch.bfloat16)
    output = torch.empty_like(hidden_states)
    cpu_infer.submit(
        moe.forward_task(
            qlen.data_ptr(),
            1,
            expert_ids.data_ptr(),
            expert_weights.data_ptr(),
            hidden_states.data_ptr(),
            output.data_ptr(),
            False,
        )
    )
    cpu_infer.sync()
    return output


def test_mxfp4_nonidentity_map_compacts_into_physical_slots():
    if not _has_cpu_flag("avx512_bf16"):
        pytest.skip("native MXFP4 AMX test requires AVX-512 BF16")

    try:
        from kt_kernel import kt_kernel_ext
    except ImportError:
        pytest.skip("kt_kernel extension is not installed")

    moe_module = kt_kernel_ext.moe
    if not hasattr(moe_module, "AMXFP4_KGroup_MOE"):
        pytest.skip("AMXFP4_KGroup_MOE is not compiled")

    hidden_size = 256
    intermediate_size = 256
    # The compact bank selects logical experts 2 and 3 from a larger pointer
    # table.  A reference bank receives those same tensors in identity order.
    sources = [
        _source_expert(nibble, hidden_size, intermediate_size)
        for nibble in (0, 1, 2, 3)
    ]
    cpu_infer = kt_kernel_ext.CPUInfer(4)
    mapped_moe, mapped_ids = _build_moe(
        moe_module, cpu_infer, sources, logical_ids=[2, 3]
    )
    selected_sources = [sources[2], sources[3]]
    reference_moe, reference_ids = _build_moe(
        moe_module, cpu_infer, selected_sources, logical_ids=[0, 1]
    )

    mapped = _forward(cpu_infer, mapped_moe, hidden_size)
    reference = _forward(cpu_infer, reference_moe, hidden_size)
    assert torch.isfinite(mapped).all()
    torch.testing.assert_close(mapped, reference, rtol=0, atol=0)

    # Keep source pointers and maps alive through both native forwards.
    assert mapped_ids.tolist() == [2, 3]
    assert reference_ids.tolist() == [0, 1]
