"""CLI para entrenamiento."""

from __future__ import annotations

from ..rl.train import train_agent


def main() -> None:
    train_agent()


if __name__ == "__main__":
    main()
