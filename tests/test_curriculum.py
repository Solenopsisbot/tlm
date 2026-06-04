from tlm.curriculum import column_add_reasoning, column_sub_reasoning, make_example


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

    assert "<answer>" in example["output"]
