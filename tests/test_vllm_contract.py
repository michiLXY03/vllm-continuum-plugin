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
        "_update_from_kv_xfer_finished",
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


def test_vllm_0251_keeps_the_async_reload_measurement_window() -> None:
    """The reload estimator needs a start point, an end point, and a size."""
    root = _vllm_source_root()
    scheduler = (root / "vllm/v1/core/sched/scheduler.py").read_text(encoding="utf-8")

    # Start: the connector reports an asynchronous load.
    assert "self.connector.get_num_new_matched_tokens(" in scheduler
    assert "if load_kv_async:" in scheduler
    assert "request.status = RequestStatus.WAITING_FOR_REMOTE_KVS" in scheduler
    # End: the worker reports the transfer finished.
    assert "self.finished_recving_kv_req_ids.add(req_id)" in scheduler
    # Size: how many tokens the tier supplied.
    assert "num_external_computed_tokens = ext_tokens" in scheduler


def test_vllm_0251_connector_returns_tokens_and_async_flag() -> None:
    methods = _class_methods(
        "vllm/distributed/kv_transfer/kv_connector/v1/base.py",
        "KVConnectorBase_V1",
    )
    signature = methods["get_num_new_matched_tokens"]
    names = [argument.arg for argument in signature.args.args]

    assert names[:3] == ["self", "request", "num_computed_tokens"]
