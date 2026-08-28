"""Generate a labeled Q2237 image-B training set on one or more GPUs.

Examples
--------
One GPU::

    python examples/generate_q2237_training_set.py --count 100 --gpus 0

Four GPUs, with one ordinary Python command::

    python examples/generate_q2237_training_set.py --count 1000 --gpus 0 1 2 3

Each spawned worker owns one CUDA device, incurs its first-call compilation
once, and then reuses the compiled kernels for every later example assigned to
that device. Existing completed NPZ files are skipped unless ``--overwrite``
is supplied.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import secrets
from pathlib import Path

import torch
from training_set_support import (
    generate_labeled_example,
    q2237_b_system,
    save_example,
)


def _worker(
    worker: int,
    device: str,
    indices: tuple[int, ...],
    args: argparse.Namespace,
) -> None:
    """Generate one deterministic shard while retaining the local compile cache."""

    output = Path(args.output_dir).resolve()
    cache_name = device.replace(":", "_")
    cache = output / "triton_cache" / cache_name
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache))
    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":", 1)[1]))
    map_times = torch.linspace(0.0, float(args.days), int(args.epochs))
    flux_times = torch.arange(
        0.0,
        float(args.days) + 0.5 * float(args.source_cadence_days),
        float(args.source_cadence_days),
    )
    for local_index, index in enumerate(indices):
        target = output / f"light_curve_{index:05d}.npz"
        if target.exists() and not args.overwrite:
            print(f"[worker {worker}] reuse {target.name}", flush=True)
            continue
        system = q2237_b_system(
            seed=args.seed + index,
            driver_seed=args.seed + 1_000_000 + index,
            device=device,
            times_days=flux_times,
            source_resolution=args.source_resolution,
            label_resolution=args.label_resolution,
        )
        result, center_magnifications, elapsed = generate_labeled_example(
            system, map_times, flux_times, rays=args.rays
        )
        saved = save_example(
            output,
            index,
            result,
            center_magnifications,
            system.source,
            system.metadata,
            runtime_seconds=elapsed,
            worker=worker,
        )
        phase = "first call (compile + execute)" if local_index == 0 else "warmed"
        print(
            f"[worker {worker} {device}] {saved.name}: {elapsed:.3f} s ({phase})",
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--gpus", type=int, nargs="*", default=None)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--output-dir", default="training_sets/q2237_b")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "base seed for exact reproducibility. When omitted, draw and print "
            "a fresh base seed that is also stored with every example"
        ),
    )
    parser.add_argument("--days", type=float, default=3650.0)
    parser.add_argument("--epochs", type=int, default=147)
    parser.add_argument("--source-cadence-days", type=float, default=1.0)
    parser.add_argument("--rays", type=int, default=10_000_000)
    parser.add_argument("--source-resolution", type=int, default=1024)
    parser.add_argument("--label-resolution", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.count < 1 or args.epochs < 2 or args.source_cadence_days <= 0.0:
        parser.error(
            "--count and --source-cadence-days must be positive and "
            "--epochs must be at least two"
        )
    return args


def main() -> None:
    """Shard the requested examples over independently pinned GPU workers."""

    args = _parse_args()
    if args.seed is None:
        args.seed = secrets.randbelow(2**63 - 1_000_000 - args.count)
        print(f"No --seed supplied. Sampled reproducible base_seed={args.seed}")
    else:
        print(f"Using explicit reproducible base_seed={args.seed}")
    if args.device == "cpu" or not torch.cuda.is_available():
        devices = ("cpu",)
    else:
        gpu_ids = args.gpus if args.gpus else list(range(torch.cuda.device_count()))
        if not gpu_ids:
            raise RuntimeError("no CUDA devices were selected")
        devices = tuple(f"cuda:{index}" for index in gpu_ids)
    shards = tuple(
        tuple(range(worker, args.count, len(devices)))
        for worker in range(len(devices))
    )
    print(f"devices={devices}, shard_sizes={tuple(map(len, shards))}")
    if len(devices) == 1:
        _worker(0, devices[0], shards[0], args)
        return
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=_worker, args=(rank, device, shards[rank], args))
        for rank, device in enumerate(devices)
        if shards[rank]
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures:
        raise RuntimeError(f"training workers failed with exit codes {failures}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
