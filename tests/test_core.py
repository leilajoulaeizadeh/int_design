from int_design import greet


def test_greet_default() -> None:
    assert greet() == "Hello, world!"


def test_greet_custom_name() -> None:
    assert greet("team") == "Hello, team!"
