"""CPU smoke test with synthetic data.

Run from the project root with ``python -m pytest -q``.
Requires pytest and PyTorch; no downloads or training are needed.
"""

import torch

from slm.ch2_simple_bigram import SimpleBigramLanguageModel, get_batch


def test_bigram_batch_loss_and_two_token_generation() -> None:
    """Check batch loss and repeatable generation of two tokens on CPU."""
    vocab_size = 4
    batch_size = 2
    block_size = 3

    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(42)
        data = torch.arange(12, dtype=torch.long, device="cpu") % vocab_size
        inputs, targets = get_batch(data, block_size, batch_size)
        model = SimpleBigramLanguageModel(vocab_size).to("cpu")

        assert inputs.shape == targets.shape == (batch_size, block_size)
        torch.testing.assert_close(targets, (inputs + 1) % vocab_size)

        logits, loss = model(inputs, targets)

        assert logits.shape == (batch_size * block_size, vocab_size)
        assert loss is not None
        assert loss.ndim == 0
        assert torch.isfinite(loss).item()
        assert loss.item() > 0
        log_probs = model.token_embedding_table(inputs).log_softmax(dim=-1)
        expected_loss = -log_probs.gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).mean()
        torch.testing.assert_close(loss, expected_loss)

        torch.manual_seed(123)
        generated = model.generate(inputs, max_new_tokens=2)
        torch.manual_seed(123)
        repeated = model.generate(inputs, max_new_tokens=2)

        assert generated.shape == (batch_size, block_size + 2)
        assert generated.device.type == "cpu"
        assert generated.dtype == torch.long
        torch.testing.assert_close(generated[:, :block_size], inputs)
        new_tokens = generated[:, block_size:]
        assert ((new_tokens >= 0) & (new_tokens < vocab_size)).all().item()
        torch.testing.assert_close(generated, repeated)
