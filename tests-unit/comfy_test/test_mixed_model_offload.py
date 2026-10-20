from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_management as mm
import comfy.memory_management as memory_management
import comfy.system_memory as system_memory


@pytest.mark.parametrize("dynamic,pins_required,keep_cache", [
    (True, 4096, True),
    (True, 0, False),
    (False, 4096, False),
])
def test_mixed_offload_keeps_dynamic_host_cache(monkeypatch, dynamic, pins_required, keep_cache):
    device = torch.device("cuda:0")
    patcher = SimpleNamespace(
        model=SimpleNamespace(),
        is_dynamic=lambda: dynamic,
        offload_device=torch.device("cpu"),
        partially_unload=Mock(return_value=512),
    )
    loaded = SimpleNamespace(
        model=patcher, device=device, currently_used=True,
        is_dead=lambda: False,
        model_offloaded_memory=lambda: 0,
        model_memory=lambda: 4096,
        model_unload=Mock(return_value=True),
    )
    monkeypatch.setattr(mm, "current_loaded_models", [loaded])
    monkeypatch.setattr(mm, "cleanup_models_gc", lambda: None)
    monkeypatch.setattr(mm, "soft_empty_cache", lambda: None)
    monkeypatch.setattr(mm, "DISABLE_SMART_MEMORY", False)
    monkeypatch.setattr(mm, "vram_state", mm.VRAMState.NORMAL_VRAM)
    monkeypatch.setattr(mm, "get_free_memory", lambda device, torch_free_too=False: (0, 0) if torch_free_too else 0)
    budget = Mock(return_value=True)
    registration = Mock(return_value=True)
    monkeypatch.setattr(mm, "ensure_pin_budget", budget)
    monkeypatch.setattr(mm, "ensure_pin_registerable", registration)

    mm.free_memory(8192, device, pins_required=pins_required)

    if keep_cache:
        patcher.partially_unload.assert_called_once_with(patcher.offload_device, 8192)
        loaded.model_unload.assert_not_called()
        assert mm.current_loaded_models == [loaded]
    else:
        patcher.partially_unload.assert_not_called()
        loaded.model_unload.assert_called_once_with(8192)
        assert mm.current_loaded_models == []
    if pins_required:
        budget.assert_called_once_with(0)
        registration.assert_called_once_with(pins_required)
    else:
        budget.assert_not_called()
        registration.assert_not_called()


def test_existing_weights_still_release_cache_under_ram_pressure(monkeypatch):
    monkeypatch.setattr(mm.args, "high_ram", False)
    monkeypatch.setattr(memory_management, "RAM_CACHE_HEADROOM", 4 * 1024 ** 3)
    monkeypatch.setattr(system_memory, "virtual_memory_available", lambda: 1024 ** 3)
    release = Mock(return_value=2 * 1024 ** 3)
    monkeypatch.setattr(mm, "free_pins", release)

    assert mm.ensure_pin_budget(0)
    requested = release.call_args.args[0]
    assert requested >= 1024 ** 3
