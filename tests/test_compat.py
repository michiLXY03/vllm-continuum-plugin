import pytest

from continuum_vllm.compat import _assert_scheduler_contract, assert_compatible_vllm


@pytest.mark.parametrize("value", ["0.25.1", "0.25.1+empty", "0.25.1+ascend"])
def test_accepts_supported_release_and_local_versions(value: str) -> None:
    assert assert_compatible_vllm(value) == value


@pytest.mark.parametrize("value", ["0.25.0", "0.26.0", "dev", "0.25"])
def test_rejects_other_or_malformed_versions(value: str) -> None:
    with pytest.raises(RuntimeError, match="supports vLLM 0.25.1"):
        assert_compatible_vllm(value)


class _CompatibleScheduler:
    add_request = schedule = _select_waiting_queue_for_scheduling = lambda self: None
    _free_request = _free_blocks = _free_request_blocks = lambda self: None
    _update_from_kv_xfer_finished = lambda self: None
    reset_prefix_cache = shutdown = lambda self: None


class _CompatibleKVCacheManager:
    def allocate_slots(self, request, num_new_tokens):
        return None


def test_accepts_expected_private_scheduler_contract() -> None:
    _assert_scheduler_contract(_CompatibleScheduler, _CompatibleKVCacheManager)


def test_rejects_missing_private_scheduler_method() -> None:
    class MissingShutdownScheduler:
        add_request = schedule = _select_waiting_queue_for_scheduling = lambda self: (
            None
        )
        _free_request = _free_blocks = _free_request_blocks = lambda self: None
        _update_from_kv_xfer_finished = lambda self: None
        reset_prefix_cache = lambda self: None

    with pytest.raises(RuntimeError, match="missing methods: shutdown"):
        _assert_scheduler_contract(MissingShutdownScheduler, _CompatibleKVCacheManager)


def test_rejects_changed_allocate_slots_signature() -> None:
    class IncompatibleKVCacheManager:
        def allocate_slots(self, sequence):
            return None

    with pytest.raises(RuntimeError, match="num_new_tokens, request"):
        _assert_scheduler_contract(_CompatibleScheduler, IncompatibleKVCacheManager)
