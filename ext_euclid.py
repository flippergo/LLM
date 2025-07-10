#!/usr/bin/env python3
"""Compute the greatest common divisor using the Extended Euclidean Algorithm.

This script provides a command line interface:

    ./ext_euclid.py <a> <b>

It prints gcd(a, b) and coefficients x, y such that a*x + b*y = gcd(a, b).
"""

from __future__ import annotations

import argparse
from typing import Tuple


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Return ``gcd(a, b)`` and integers ``x`` and ``y`` satisfying
    ``a*x + b*y == gcd(a, b)``.
    The gcd is always non-negative.
    """
    if b == 0:
        gcd = abs(a)
        x = 1 if a >= 0 else -1
        return gcd, x, 0
    gcd, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return gcd, x, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute gcd and Bézout coefficients using the extended Euclidean algorithm."
    )
    parser.add_argument("a", type=int, help="First integer")
    parser.add_argument("b", type=int, help="Second integer")
    args = parser.parse_args()

    gcd, x, y = extended_gcd(args.a, args.b)
    print(f"gcd({args.a}, {args.b}) = {gcd}")
    print(f"coefficients: x = {x}, y = {y}")
    print(f"verification: {args.a}*{x} + {args.b}*{y} = {args.a * x + args.b * y}")


if __name__ == "__main__":
    main()
