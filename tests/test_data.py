import torch

from tlm.data import TextCodec, build_codec


def test_bpe_codec_round_trip() -> None:
    text = "ROMEO:\nThe stream remembers the shape of things.\n"
    codec = build_codec(text, mode="bpe", vocab_size=512)
    tokens = codec.encode(text)

    assert isinstance(tokens, torch.Tensor)
    assert codec.decode(tokens) == text
    assert codec.vocab_size <= 512


def test_bpe_codec_serializes() -> None:
    text = "JULIET:\nA tiny model may still dream in tokens.\n"
    codec = build_codec(text, mode="bpe", vocab_size=512)
    restored = TextCodec.from_dict(codec.to_dict())

    assert restored.decode(restored.encode(text)) == text
