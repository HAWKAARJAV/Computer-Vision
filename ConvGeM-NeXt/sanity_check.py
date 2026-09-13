"""
sanity_check.py — Dummy forward / loss / backward for all four variants.

Usage:
    python sanity_check.py
    python sanity_check.py --num_classes 5 --pretrained
"""

import argparse

from model import run_architecture_sanity_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Architecture sanity check.")
    parser.add_argument("--num_classes", type=int, default=38)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_architecture_sanity_check(
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        debug=not args.no_debug,
    )
