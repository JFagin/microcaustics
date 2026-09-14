"""Execute tutorial notebooks and save outputs only after successful runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PACKAGE_ROOT / "examples" / "notebooks"
BUILD_ROOT = PACKAGE_ROOT / "build" / "notebook_execution"


def _configure_environment() -> None:
    """Keep Jupyter and compiler caches inside ignored package directories."""

    jupyter_root = PACKAGE_ROOT / ".jupyter"
    cache_root = BUILD_ROOT / "compiler_cache"
    jupyter_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_CONFIG_DIR"] = str(jupyter_root)
    os.environ["JUPYTER_DATA_DIR"] = str(jupyter_root / "data")
    os.environ["IPYTHONDIR"] = str(jupyter_root)
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
    source_root = str(PACKAGE_ROOT / "src")
    existing = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        source_root if not existing else os.pathsep.join((source_root, existing))
    )


def execute_notebook(path: Path, *, timeout: int, kernel_name: str) -> float:
    """Run beside the notebook, retaining media, and atomically save cell outputs."""

    import nbformat
    from nbclient import NotebookClient

    started = perf_counter()
    notebook = nbformat.read(path, as_version=4)
    executed = NotebookClient(
        notebook,
        timeout=int(timeout),
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(path.parent.resolve())}},
        allow_errors=False,
    ).execute()
    temporary_path = path.with_suffix(".ipynb.tmp")
    nbformat.write(executed, temporary_path)
    temporary_path.replace(path)
    return perf_counter() - started


def main() -> None:
    """Execute all notebooks, or only names selected on the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Notebook names or stems")
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds per cell")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name")
    args = parser.parse_args()

    _configure_environment()
    discovered = sorted(
        path for path in NOTEBOOK_ROOT.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    )
    if not discovered:
        raise SystemExit(f"No notebooks found in {NOTEBOOK_ROOT}")
    available = {path.name: path for path in discovered}
    available.update({path.stem: path for path in available.values()})
    available.update({path.relative_to(NOTEBOOK_ROOT).as_posix(): path for path in discovered})
    if args.notebooks:
        missing = [name for name in args.notebooks if name not in available]
        if missing:
            raise SystemExit(f"unknown notebooks: {', '.join(missing)}")
        paths = [available[name] for name in args.notebooks]
    else:
        paths = discovered

    total_started = perf_counter()
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] executing {path.name}", flush=True)
        elapsed = execute_notebook(
            path,
            timeout=args.timeout,
            kernel_name=args.kernel,
        )
        print(f"[{index}/{len(paths)}] completed {path.name} in {elapsed:.1f} s", flush=True)
    print(
        f"Executed {len(paths)} notebooks in {perf_counter() - total_started:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
