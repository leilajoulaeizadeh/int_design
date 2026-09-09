"""CLI entrypoint for int_design."""

from .core import greet


def main() -> None:
    """Run a simple CLI greeting."""
    print(greet())


if __name__ == "__main__":
    main()
