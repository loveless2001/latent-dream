import pytest
import torch

from kv_dreaming.diagnostic import continuation_kl


@pytest.mark.parametrize("continuation", [1, 5])
@torch.inference_mode()
def test_diagnostic_keeps_aligned_prediction_rows(runtime, monkeypatch, continuation):
    ids = [3, 8, 13, 6, 17, 24, 19, 28, 41, 29, 11, 22, 34, 9, 12, 5, 7]
    prefix = 12
    ids = ids[:prefix + continuation]
    forward = runtime.model.forward
    dense = forward(torch.tensor([ids]), use_cache=False).logits
    expected = dense[:, prefix - 1:prefix + continuation - 1]
    calls = []

    def checked_forward(*args, **kwargs):
        assert kwargs.get("logits_to_keep") == continuation + 1
        output = forward(*args, **kwargs)
        assert output.logits.shape[1] == continuation + 1
        torch.testing.assert_close(output.logits[:, :continuation], expected)
        calls.append(True)
        return output

    monkeypatch.setattr(runtime.model, "forward", checked_forward)
    report = continuation_kl(runtime, ids, prefix=prefix, continuation=continuation,
                             budget=32, recent=4)
    assert len(calls) == 1
    for result in report["policies"].values():
        assert len(result["per_token_kl"]) == continuation
        assert result["reduction_events"] == 0
        assert max(abs(value) for value in result["per_token_kl"]) < 1e-6
