import torch
from safetensors.torch import save_file

from kt_kernel.utils.loader import NVFP4SafeTensorLoader


def _e4m3(values):
    return torch.tensor(values, dtype=torch.float32).to(torch.float8_e4m3fn)


def test_modelopt_nvfp4_loader_preserves_native_tensors(tmp_path):
    tensors = {}
    prefix = "model.layers.0.mlp.experts"
    for expert_id in range(2):
        for projection, rows, columns in (
            ("gate_proj", 4, 16),
            ("up_proj", 4, 16),
            ("down_proj", 8, 16),
        ):
            key = f"{prefix}.{expert_id}.{projection}"
            tensors[f"{key}.weight"] = torch.arange(
                rows * columns // 2, dtype=torch.uint8
            ).reshape(rows, columns // 2)
            tensors[f"{key}.weight_scale"] = _e4m3(
                [0.5 + expert_id] * (rows * columns // 16)
            ).reshape(rows, columns // 16)
            tensors[f"{key}.weight_scale_2"] = torch.tensor(
                0.25 + expert_id, dtype=torch.float32
            )

    save_file(tensors, tmp_path / "model.safetensors")
    loader = NVFP4SafeTensorLoader(str(tmp_path))
    result = loader.load_experts("model.layers.0")

    assert len(result["gate"]) == 2
    assert result["gate"][0].dtype == torch.uint8
    assert result["gate"][0].shape == (4, 8)
    assert result["gate_scale"][0].dtype == torch.float8_e4m3fn
    assert result["gate_scale"][0].shape == (4, 1)
    assert result["gate_scale_2"][0].dtype == torch.float32
    assert result["gate_scale_2"][0].shape == (1,)
    assert result["gate_scale_2"][1].item() == 1.25
    assert result["down"][0].shape == (8, 8)


def test_modelopt_nvfp4_loader_probes_stripped_model_prefix(tmp_path):
    tensors = {}
    prefix = "layers.3.mlp.experts.0"
    for projection in ("gate_proj", "up_proj", "down_proj"):
        key = f"{prefix}.{projection}"
        tensors[f"{key}.weight"] = torch.zeros((2, 8), dtype=torch.uint8)
        tensors[f"{key}.weight_scale"] = _e4m3([1.0, 1.0]).reshape(2, 1)
        tensors[f"{key}.weight_scale_2"] = torch.ones((), dtype=torch.float32)

    save_file(tensors, tmp_path / "model.safetensors")
    loader = NVFP4SafeTensorLoader(str(tmp_path))
    result = loader.load_experts("model.layers.3")

    assert len(result["up"]) == 1
    assert result["up_scale_2"][0].item() == 1.0


def test_modelopt_nvfp4_loader_reads_only_selected_experts(tmp_path):
    tensors = {}
    prefix = "model.layers.0.mlp.experts"
    for expert_id in range(3):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            key = f"{prefix}.{expert_id}.{projection}"
            tensors[f"{key}.weight"] = torch.full((2, 8), expert_id, dtype=torch.uint8)
            tensors[f"{key}.weight_scale"] = _e4m3([1.0, 1.0]).reshape(2, 1)
            tensors[f"{key}.weight_scale_2"] = torch.ones((), dtype=torch.float32)

    save_file(tensors, tmp_path / "model.safetensors")
    loader = NVFP4SafeTensorLoader(str(tmp_path))
    result = loader.load_experts("model.layers.0", expert_ids=[0, 2])

    assert result["gate"][0] is not None
    assert result["gate"][1] is None
    assert result["gate"][2] is not None
    assert result["gate_scale_2"][1] is None
