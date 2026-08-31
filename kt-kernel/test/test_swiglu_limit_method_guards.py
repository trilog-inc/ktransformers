"""Regression guards for native-MoE ``swiglu_limit`` method dispatch.

The tests execute the relevant source AST in isolation so they do not require a
locally compiled ``kt_kernel_ext``.  Real-extension coverage lives in the GLM
qj5090 acceptance oracle.
"""

from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace


KT_KERNEL_ROOT = Path(__file__).resolve().parents[1]
EXPERTS_PATH = KT_KERNEL_ROOT / "python/experts.py"
AMX_PATH = KT_KERNEL_ROOT / "python/utils/amx.py"
ALLOWED_METHODS = ("FP8", "MXFP4", "NVFP4", "MXFP8")
REJECTED_METHODS = (
    "RAWINT4",
    "BF16",
    "FP8_PERCHANNEL",
    "GPTQ_INT4",
    "SYCL_GPTQ_INT4",
)


class _Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _NativeRecorder(_Recorder):
    pass


class _AMXRecorder(_Recorder):
    pass


class _LlamafileRecorder(_Recorder):
    pass


class _GeneralRecorder(_Recorder):
    pass


def _compile_factory():
    tree = ast.parse(EXPERTS_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_create_inference_wrapper"
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                copy.deepcopy(function),
            ],
            type_ignores=[],
        )
    )
    namespace = {
        "AMXMoEWrapper": _AMXRecorder,
        "NativeMoEWrapper": _NativeRecorder,
        "LlamafileMoEWrapper": _LlamafileRecorder,
        "GeneralMoEWrapper": _GeneralRecorder,
    }
    exec(compile(module, str(EXPERTS_PATH), "exec"), namespace)
    return namespace["_create_inference_wrapper"]


def _native_guard(method_name: str):
    tree = ast.parse(AMX_PATH.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NativeMoEWrapper"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    guard = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and "swiglu_limit" in ast.unparse(node.test)
        and "not in" in ast.unparse(node.test)
    )
    if method_name == "__init__":
        args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="swiglu_limit"), ast.arg(arg="method")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        )
    else:
        args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        )
    wrapper = ast.FunctionDef(
        name="guard",
        args=args,
        body=[copy.deepcopy(guard)],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(AMX_PATH), "exec"), namespace)
    return namespace["guard"]


def _factory_kwargs(method: str, limit: float):
    return dict(
        layer_idx=0,
        num_experts=1,
        num_experts_per_tok=1,
        hidden_size=128,
        moe_intermediate_size=128,
        gpu_experts_mask=None,
        cpuinfer_threads=1,
        threadpool_count=1,
        weight_path="unused",
        chunked_prefill_size=1,
        cpu_save=False,
        max_deferred_experts_per_token=None,
        method=method,
        numa_nodes=None,
        swiglu_limit=limit,
        swiglu_alpha=0.0,
    )


class TestSwigluLimitMethodGuards(unittest.TestCase):
    def test_factory_forwards_limit_to_block_fp8(self):
        wrapper = _compile_factory()(**_factory_kwargs("FP8", 10.0))

        self.assertEqual(wrapper.kwargs["method"], "FP8")
        self.assertEqual(wrapper.kwargs["swiglu_limit"], 10.0)
        self.assertEqual(wrapper.kwargs["swiglu_alpha"], 0.0)

    def test_factory_does_not_widen_other_native_formats(self):
        factory = _compile_factory()
        for method in REJECTED_METHODS:
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "only supported"):
                    factory(**_factory_kwargs(method, 10.0))

    def test_factory_forwards_bounded_native_loader_options(self):
        kwargs = _factory_kwargs("FP8", 10.0)
        kwargs.update(
            weight_base_key="model.layers.45",
            release_loader_after_load=False,
        )

        wrapper = _compile_factory()(**kwargs)

        self.assertEqual(wrapper.kwargs["weight_base_key"], "model.layers.45")
        self.assertFalse(wrapper.kwargs["release_loader_after_load"])

    def test_factory_rejects_bounded_loader_options_for_non_native_backend(self):
        factory = _compile_factory()
        for option in (
            {"weight_base_key": "model.layers.45"},
            {"release_loader_after_load": False},
        ):
            with self.subTest(option=option):
                kwargs = _factory_kwargs("AMXINT4", 0.0)
                kwargs.update(option)
                with self.assertRaisesRegex(ValueError, "native SafeTensor"):
                    factory(**kwargs)

    def test_both_native_wrapper_guards_have_the_same_exact_allow_list(self):
        init_guard = _native_guard("__init__")
        load_guard = _native_guard("load_weights")

        for method in ALLOWED_METHODS:
            with self.subTest(guard="init", method=method):
                init_guard(10.0, method)
            with self.subTest(guard="load", method=method):
                load_guard(SimpleNamespace(swiglu_limit=10.0, method=method))

        for method in REJECTED_METHODS:
            with self.subTest(guard="init", method=method):
                with self.assertRaisesRegex(ValueError, "supported only"):
                    init_guard(10.0, method)
            with self.subTest(guard="load", method=method):
                with self.assertRaisesRegex(ValueError, "only valid"):
                    load_guard(SimpleNamespace(swiglu_limit=10.0, method=method))

    def test_native_loader_uses_exact_prefix_and_bounded_release_switch(self):
        tree = ast.parse(AMX_PATH.read_text(encoding="utf-8"))
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NativeMoEWrapper"
        )
        load_weights = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "load_weights"
        )
        source = ast.unparse(load_weights)

        self.assertIn("[self.weight_base_key]", source)
        self.assertIn("if self.release_loader_after_load", source)
        self.assertIn("_release_loader(layer_idx=self.layer_idx)", source)

    def test_native_loader_forwards_activation_parameters_to_cpp(self):
        tree = ast.parse(AMX_PATH.read_text(encoding="utf-8"))
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NativeMoEWrapper"
        )
        load_weights = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "load_weights"
        )
        source = ast.unparse(load_weights)

        self.assertIn("moe_config.swiglu_limit = self.swiglu_limit", source)
        self.assertIn("moe_config.swiglu_alpha = self._swiglu_alpha", source)


if __name__ == "__main__":
    unittest.main()
