"""Reproducible independent random streams from one user-facing seed."""

from __future__ import annotations

import hashlib


def derive_seed(seed: int | None, component: str) -> int | None:
    """Derive a stable independent 63-bit seed for ``component``.

    A missing base seed intentionally remains nondeterministic. The mapping is
    stable across Python processes and does not depend on Python's randomized
    string hash.
    """

    if seed is None:
        return None
    label = str(component).strip()
    if not label:
        raise ValueError("component must be non-empty")
    payload = f"microcaustics:{int(seed)}:{label}".encode()
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "little"
    ) & (2**63 - 1)
