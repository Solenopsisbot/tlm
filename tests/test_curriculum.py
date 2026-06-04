from tlm.curriculum import (
    column_add_reasoning,
    column_sub_reasoning,
    make_example,
    structured_add_reasoning,
    structured_sub_reasoning,
)


def test_column_add_reasoning_includes_carry() -> None:
    text = column_add_reasoning(57, 63, 120)

    assert "7 + 3 = 10" in text
    assert "carry 1" in text
    assert "5 + 6 + 1 = 12" in text
    assert "57 + 63 = 120" in text


def test_column_sub_reasoning_includes_borrow() -> None:
    text = column_sub_reasoning(42, 17, 25)

    assert "borrow 1 ten" in text
    assert "12 - 7 = 5" in text
    assert "3 - 1 = 2" in text
    assert "42 - 17 = 25" in text


def test_make_example_column_style() -> None:
    example = make_example(
        __import__("random").Random(1),
        stage="addsub_2digit",
        chain_of_thought=True,
        reasoning_style="column",
    )

    assert "</think>" in example["output"]


def test_structured_add_reasoning_has_bindings() -> None:
    text = structured_add_reasoning(93, 98, 191)

    assert "problem = 93 + 98" in text
    assert "a_ones = 3" in text
    assert "b_ones = 8" in text
    assert "carry = 1" in text
    assert "result = 191" in text


def test_structured_sub_reasoning_has_bindings() -> None:
    text = structured_sub_reasoning(42, 17, 25)

    assert "problem = 42 - 17" in text
    assert "borrow = 1" in text
    assert "adjusted_ones = a_ones + 10 * borrow = 2 + 10 * 1 = 12" in text
    assert "tens_digit = adjusted_tens - b_tens = 3 - 1 = 2" in text
    assert "result = 25" in text


def test_make_example_can_reflect() -> None:
    example = make_example(
        __import__("random").Random(1),
        stage="addsub_2digit",
        chain_of_thought=True,
        reasoning_style="structured",
        reflect=True,
    )

    assert "<reflect>" in example["output"]
