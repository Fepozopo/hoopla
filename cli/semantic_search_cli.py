#!/usr/bin/env python3
from pathlib import Path
import sys

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    args = parser.parse_args()

    match args.command:
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
