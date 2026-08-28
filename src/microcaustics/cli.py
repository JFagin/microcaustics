"""Command-line utilities for installation and runtime diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .runtime import RuntimeCapabilities, resolve_runtime


def _doctor() -> int:
    capabilities = RuntimeCapabilities.detect()
    payload = asdict(capabilities)
    memory = payload.get("total_device_memory_bytes")
    if memory is not None:
        payload["total_device_memory_gib"] = memory / 1024**3
    resolved = resolve_runtime(capabilities=capabilities)
    payload["automatic_selection"] = {
        "device": str(resolved.device),
        "backend": resolved.backend.value,
        "dtype": str(resolved.dtype).replace("torch.", ""),
        "memory_fraction": resolved.memory_fraction,
        "torch_compile_mode": resolved.torch_compile_mode,
        "memory_budget_gib": (
            None
            if resolved.available_memory_bytes is None
            else resolved.available_memory_bytes / 1024**3
        ),
        "fallback_reason": resolved.fallback_reason,
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the ``microcaustics`` command-line interface."""

    parser = argparse.ArgumentParser(prog="microcaustics")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor",
        help="report available devices, PyTorch compilation, and Triton support",
    )
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
