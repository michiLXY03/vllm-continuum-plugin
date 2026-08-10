import ast
import os
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def _vllm_source_root() -> Path:
    configured_root = os.environ.get("CONTINUUM_VLLM_SOURCE_ROOT")
    if configured_root:
        return Path(configured_root)

    try:
        return Path(distribution("vllm").locate_file(""))
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "install vLLM or set CONTINUUM_VLLM_SOURCE_ROOT to its checkout"
        ) from exc


def _class_methods(path: str, class_name: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse((_vllm_source_root() / path).read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    return {
        item.name: item
        for item in node.body
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def test_vllm_0251_scheduler_keeps_required_override_points() -> None:
    methods = _class_methods("vllm/v1/core/sched/scheduler.py", "Scheduler")

    assert {
        "add_request",
        "schedule",
        "_select_waiting_queue_for_scheduling",
        "_free_request",
        "_free_blocks",
        "_free_request_blocks",
        "reset_prefix_cache",
        "shutdown",
    } <= methods.keys()


def test_vllm_0251_allocate_slots_contract_accepts_request_and_kwargs() -> None:
    methods = _class_methods("vllm/v1/core/kv_cache_manager.py", "KVCacheManager")
    allocate_slots = methods["allocate_slots"]
    names = [argument.arg for argument in allocate_slots.args.args]

    assert names[:3] == ["self", "request", "num_new_tokens"]
    assert {
        "new_computed_blocks",
        "num_lookahead_tokens",
        "delay_cache_blocks",
        "full_sequence_must_fit",
    } <= set(names)


def test_vllm_0251_exposes_custom_scheduler_and_request_xargs() -> None:
    root = _vllm_source_root()
    scheduler_config = (root / "vllm/config/scheduler.py").read_text(encoding="utf-8")
    assert "scheduler_cls:" in scheduler_config
    assert "resolve_obj_by_qualname(self.scheduler_cls)" in scheduler_config

    protocols = (
        "vllm/entrypoints/openai/chat_completion/protocol.py",
        "vllm/entrypoints/openai/completion/protocol.py",
        "vllm/entrypoints/openai/responses/protocol.py",
    )
    for path in protocols:
        source = (root / path).read_text(encoding="utf-8")
        assert "vllm_xargs:" in source
        assert "extra_args=extra_args" in source
