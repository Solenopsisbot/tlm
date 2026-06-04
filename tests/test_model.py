import torch

from tlm.model import TinyLanguageModel, TinyLanguageModelConfig, should_stop
from tlm.sample import config_from_checkpoint


def test_forward_shape() -> None:
    model = TinyLanguageModel(
        TinyLanguageModelConfig(
            vocab_size=65,
            dim=32,
            layers=1,
            architecture="rnn",
            recurrent="gated",
        )
    )
    x = torch.randint(0, 65, (4, 16))
    logits, state = model(x)

    assert logits.shape == (4, 16, 65)
    assert len(state) == 1
    assert state[0].shape == (4, 32)


def test_generate_adds_tokens() -> None:
    model = TinyLanguageModel(TinyLanguageModelConfig(vocab_size=10, dim=32, layers=1))
    prompt = torch.tensor([1, 2, 3])
    generated = model.generate(prompt, max_new_tokens=5)

    assert generated.shape == (8,)


def test_swift_forward_shape() -> None:
    model = TinyLanguageModel(
        TinyLanguageModelConfig(
            vocab_size=65,
            dim=32,
            layers=2,
            architecture="swift",
            state_heads=4,
            state_rank=8,
        )
    )
    x = torch.randint(0, 65, (4, 16))
    logits, state = model(x)

    assert logits.shape == (4, 16, 65)
    assert state is None


def test_stream_forward_shape() -> None:
    model = TinyLanguageModel(
        TinyLanguageModelConfig(
            vocab_size=65,
            dim=32,
            layers=2,
            architecture="stream",
        )
    )
    x = torch.randint(0, 65, (4, 16))
    logits, state = model(x)

    assert logits.shape == (4, 16, 65)
    assert state is None


def test_conv_forward_shape() -> None:
    model = TinyLanguageModel(
        TinyLanguageModelConfig(
            vocab_size=65,
            dim=32,
            layers=4,
            architecture="conv",
            state_kernel=3,
            max_dilation=8,
        )
    )
    x = torch.randint(0, 65, (4, 16))
    logits, state = model(x)

    assert logits.shape == (4, 16, 65)
    assert state is None


def test_conv_generate_adds_tokens() -> None:
    model = TinyLanguageModel(
        TinyLanguageModelConfig(vocab_size=10, dim=32, layers=2, architecture="conv")
    )
    prompt = torch.tensor([1, 2, 3])
    generated = model.generate(prompt, max_new_tokens=5)

    assert generated.shape == (8,)


def test_gru_forward_shape() -> None:
    model = TinyLanguageModel(
        TinyLanguageModelConfig(
            vocab_size=65,
            dim=32,
            layers=1,
            architecture="rnn",
            recurrent="gru",
        )
    )
    x = torch.randint(0, 65, (4, 16))
    logits, state = model(x)

    assert logits.shape == (4, 16, 65)
    assert state.shape == (1, 4, 32)


def test_legacy_checkpoint_config_defaults_to_rnn() -> None:
    config = config_from_checkpoint({"vocab_size": 65, "dim": 32, "layers": 1})

    assert config.architecture == "rnn"


def test_generate_stops_on_stop_sequence() -> None:
    assert should_stop(torch.tensor([1, 2, 3, 4]), [torch.tensor([3, 4])])
