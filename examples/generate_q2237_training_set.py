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
from pathlib import Path

import torch
from training_set_support import (
    generate_labeled_example,
    generate_labeled_examples,
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
    pending = tuple(
        index
        for index in indices
        if args.overwrite or not (output / f"light_curve_{index:05d}.npz").exists()
    )
    for index in indices:
        target = output / f"light_curve_{index:05d}.npz"
        if index not in pending:
            print(f"[worker {worker}] reuse {target.name}", flush=True)
    if not pending:
        return

    # Compile once before launching independent systems concurrently. Concurrent
    # first-time compilation is slower and can race on compiler caches.
    first_index = pending[0]
    first_system = q2237_b_system(
        seed=args.seed + first_index,
        driver_seed=args.seed + 1_000_000 + first_index,
        device=device,
        times_days=flux_times,
        source_resolution=args.source_resolution,
        label_resolution=args.label_resolution,
    )
    first_result, first_centers, first_elapsed = generate_labeled_example(
        first_system, map_times, flux_times, rays=args.rays
    )
    saved = save_example(
        output,
        first_index,
        first_result,
        first_centers,
        first_system.source,
        first_system.metadata,
        runtime_seconds=first_elapsed,
        worker=worker,
    )
    print(
        f"[worker {worker} {device}] {saved.name}: {first_elapsed:.3f} s "
        "(first call, compile + execute)",
        flush=True,
    )

    remaining = pending[1:]
    for start in range(0, len(remaining), args.curves_per_batch):
        group_indices = remaining[start : start + args.curves_per_batch]
        group_systems = tuple(
            q2237_b_system(
                seed=args.seed + index,
                driver_seed=args.seed + 1_000_000 + index,
                device=device,
                times_days=flux_times,
                source_resolution=args.source_resolution,
                label_resolution=args.label_resolution,
            )
            for index in group_indices
        )
        generated, batch_report = generate_labeled_examples(
            group_systems,
            map_times,
            flux_times,
            rays=args.rays,
            curves_per_batch=args.curves_per_batch,
        )
        for index, system, (result, centers, elapsed) in zip(
            group_indices, group_systems, generated, strict=True
        ):
            metadata = {
                **system.metadata,
                "requested_curves_per_batch": batch_report.requested_curves_per_batch,
                "executed_curve_batch_sizes": batch_report.executed_batch_sizes,
                "curve_batch_oom_reductions": batch_report.oom_reductions,
            }
            saved = save_example(
                output,
                index,
                result,
                centers,
                system.source,
                metadata,
                runtime_seconds=elapsed,
                worker=worker,
            )
            print(
                f"[worker {worker} {device}] {saved.name}: {elapsed:.3f} s "
                f"(warmed, curves_per_batch={args.curves_per_batch})",
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
        default=0,
        help="reproducible base seed (default 0); example indices select new realizations",
    )
    parser.add_argument("--days", type=float, default=3650.0)
    parser.add_argument("--epochs", type=int, default=147)
    parser.add_argument("--source-cadence-days", type=float, default=1.0)
    parser.add_argument("--rays", type=int, default=10_000_000)
    parser.add_argument("--source-resolution", type=int, default=1024)
    parser.add_argument("--label-resolution", type=int, default=8192)
    parser.add_argument(
        "--curves-per-batch",
        type=int,
        default=1,
        help="independent systems evaluated concurrently on each GPU",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (
        args.count < 1
        or args.epochs < 2
        or args.source_cadence_days <= 0.0
        or args.curves_per_batch < 1
    ):
        parser.error(
            "--count, --source-cadence-days, and --curves-per-batch must be "
            "positive and --epochs must be at least two"
        )
    return args


def main() -> None:
    """Shard the requested examples over independently pinned GPU workers."""

    args = _parse_args()
    print(f"Using reproducible base_seed={args.seed}")
    if args.device == "cpu" or not torch.cuda.is_available():
        devices = ("cpu",)
    else:
        gpu_ids = args.gpus if args.gpus else list(range(torch.cuda.device_count()))
        if not gpu_ids:
            raise RuntimeError("no CUDA devices were selected")
        devices = tuple(f"cuda:{index}" for index in gpu_ids)
    shards = tuple(
        tuple(range(worker, args.count, len(devices))) for worker in range(len(devices))
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
