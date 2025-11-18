#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    parser.add_subparsers(dest="command", help="Available commands")

    args = parser.parse_args()

    match args.command:
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
