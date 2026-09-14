"""Cross-platform runtime discovery and safe backend selection."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import Backend, ProfilingLevel, RuntimeConfig

_WARNED_BACKEND_FALLBACKS: set[tuple[str, type[BaseException], str]] = set()
_WINDOWS_TOOLCHAIN_LOCK = threading.Lock()
_WINDOWS_TOOLCHAIN_PROBED = False
_COMPILATION_WARNING_LOCK = threading.Lock()
_COMPILATION_WARNING_EVENT = 0


class CompilationWarning(UserWarning):
    """A package-managed compiled kernel is being prepared for first use."""


def warn_compilation(
    component: str,
    *,
    backend: str,
    device: torch.device | str,
    dtype: torch.dtype,
    enabled: bool,
) -> None:
    """Report a potentially slow first call or new compiled specialization."""

    if not enabled:
        return
    global _COMPILATION_WARNING_EVENT
    # Compilation may begin on several per-curve worker threads. Serialize the
    # counter so each warning has a unique, monotonically increasing event ID.
    with _COMPILATION_WARNING_LOCK:
        _COMPILATION_WARNING_EVENT += 1
        event = _COMPILATION_WARNING_EVENT
    warnings.warn(
        f"Compilation event {event}: preparing a new {backend} specialization "
        f"for {component} on "
        f"{device} ({dtype}). This first call may be slow while code is "
        "compiled or loaded from the compiler cache. Another warning for "
        "this component indicates a new specialization or cache miss. Set "
        "warn_on_compile=False to suppress these messages.",
        CompilationWarning,
        stacklevel=3,
    )


def _prepend_environment_paths(name: str, paths: list[Path]) -> None:
    """Prepend existing paths to one process-local search variable."""

    additions = [str(path) for path in paths if path.exists()]
    if not additions:
        return
    current = os.environ.get(name, "")
    os.environ[name] = os.pathsep.join(additions + ([current] if current else []))


def _activate_installed_windows_toolchain() -> bool:
    """Expose an installed MSVC/Windows SDK to this Python process.

    Visual Studio Build Tools deliberately does not add ``cl.exe`` to the
    global PATH. Notebook kernels and ordinary Python scripts are consequently
    not developer shells. For Inductor, populate the same minimal PATH,
    INCLUDE, and LIB search roots without changing the user's global
    environment. The operation is idempotent and Windows-only.
    """

    global _WINDOWS_TOOLCHAIN_PROBED
    if platform.system() != "Windows":
        return True
    if shutil.which("cl.exe") is not None:
        return True
    with _WINDOWS_TOOLCHAIN_LOCK:
        if shutil.which("cl.exe") is not None:
            return True
        if _WINDOWS_TOOLCHAIN_PROBED:
            return False
        _WINDOWS_TOOLCHAIN_PROBED = True

        program_roots = [
            Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")),
            Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")),
        ]
        compiler_candidates: list[Path] = []
        for root in program_roots:
            compiler_candidates.extend(
                root.glob(
                    "Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"
                )
            )
        if not compiler_candidates:
            return False
        compiler = max(compiler_candidates, key=lambda path: str(path.parent))
        msvc_root = compiler.parents[3]

        sdk_root = program_roots[0] / "Windows Kits" / "10"
        sdk_candidates = [
            path
            for path in (sdk_root / "Include").glob("*")
            if path.is_dir() and (sdk_root / "Lib" / path.name).is_dir()
        ]
        if not sdk_candidates:
            return False
        sdk_include = max(sdk_candidates, key=lambda path: path.name)
        sdk_version = sdk_include.name

        _prepend_environment_paths(
            "PATH",
            [compiler.parent, sdk_root / "bin" / sdk_version / "x64"],
        )
        _prepend_environment_paths(
            "INCLUDE",
            [
                msvc_root / "include",
                sdk_include / "ucrt",
                sdk_include / "shared",
                sdk_include / "um",
                sdk_include / "winrt",
                sdk_include / "cppwinrt",
            ],
        )
        _prepend_environment_paths(
            "LIB",
            [
                msvc_root / "lib" / "x64",
                sdk_root / "Lib" / sdk_version / "ucrt" / "x64",
                sdk_root / "Lib" / sdk_version / "um" / "x64",
            ],
        )
        os.environ.setdefault("VCToolsInstallDir", f"{msvc_root}{os.sep}")
        os.environ.setdefault("WindowsSdkDir", f"{sdk_root}{os.sep}")
        os.environ.setdefault("WindowsSDKVersion", f"{sdk_version}{os.sep}")
        return shutil.which("cl.exe") is not None


def warn_backend_fallback(component: str, error: BaseException) -> None:
    """Warn once when an accelerated operation falls back to portable Torch.

    Automatic fallback is intentionally retained for portability. Emitting a
    warning prevents kernel defects from being mistaken for accelerated
    timings. Reproducible benchmarks should additionally set
    ``RuntimeConfig(strict_backend=True)``.
    """

    key = (str(component), type(error), str(error))
    if key in _WARNED_BACKEND_FALLBACKS:
        return
    _WARNED_BACKEND_FALLBACKS.add(key)
    warnings.warn(
        f"{component} failed with {type(error).__name__}: {error}. "
        "using the portable Torch implementation",
        RuntimeWarning,
        stacklevel=2,
    )


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Hardware and software capabilities visible to the current process."""

    platform: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    mps_available: bool
    torch_compile_available: bool
    triton_importable: bool
    gpu_name: str | None
    total_device_memory_bytes: int | None

    @classmethod
    def detect(cls) -> RuntimeCapabilities:
        """Inspect PyTorch and optional Triton without compiling any kernels."""

        cuda = bool(torch.cuda.is_available())
        mps_backend = getattr(torch.backends, "mps", None)
        mps = bool(mps_backend is not None and mps_backend.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda else None
        total_memory = (
            int(torch.cuda.get_device_properties(0).total_memory) if cuda else None
        )
        return cls(
            platform=platform.platform(),
            torch_version=str(torch.__version__),
            cuda_available=cuda,
            cuda_version=None
            if torch.version.cuda is None
            else str(torch.version.cuda),
            mps_available=mps,
            torch_compile_available=callable(getattr(torch, "compile", None)),
            triton_importable=importlib.util.find_spec("triton") is not None,
            gpu_name=gpu_name,
            total_device_memory_bytes=total_memory,
        )


@dataclass(frozen=True)
class ResolvedRuntime:
    """Concrete runtime selected from a :class:`RuntimeConfig`."""

    device: torch.device
    backend: Backend
    dtype: torch.dtype
    memory_fraction: float
    strict_backend: bool
    torch_compile_mode: str | None
    warn_on_compile: bool
    profiling: ProfilingLevel
    capabilities: RuntimeCapabilities
    fallback_reason: str | None = None

    @property
    def available_memory_bytes(self) -> int | None:
        """Return the package memory budget for a CUDA device, if known."""

        total = self.capabilities.total_device_memory_bytes
        return None if total is None else int(total * self.memory_fraction)

    @property
    def profiling_enabled(self) -> bool:
        """Whether this runtime collects synchronized timing metadata."""

        return self.profiling is not ProfilingLevel.OFF

    def synchronize(
        self,
        *,
        detailed: bool = True,
        force: bool = False,
    ) -> None:
        """Wait for queued accelerator work when timing requires it.

        Ordinary production calls default to ``profiling='off'`` and skip
        timing-only barriers. ``profiling='total'`` enables only calls marked
        ``detailed=False``. Autotuners may request an unconditional barrier
        with ``force=True``.
        """

        if not force:
            if self.profiling is ProfilingLevel.OFF:
                return
            if detailed and self.profiling is not ProfilingLevel.DETAILED:
                return

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()


def _auto_device(capabilities: RuntimeCapabilities) -> torch.device:
    if capabilities.cuda_available:
        return torch.device("cuda")
    if capabilities.mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def _torch_compile_supported(
    device: torch.device,
    capabilities: RuntimeCapabilities,
) -> bool:
    """Whether the selected device has a usable Inductor toolchain."""

    if not capabilities.torch_compile_available:
        return False
    # Inductor emits a native C++ wrapper on Windows for both CPU and CUDA
    # graphs.  Merely having CUDA/Triton available therefore does not make
    # ``torch.compile`` usable: a host compiler must also be discoverable.
    if platform.system() != "Windows":
        return True
    _activate_installed_windows_toolchain()
    configured = os.environ.get("CXX")
    if configured:
        compiler = Path(configured)
        if compiler.exists() or shutil.which(configured):
            return True
    return any(
        shutil.which(candidate) is not None
        for candidate in ("cl.exe", "clang++.exe", "g++.exe")
    )


def resolve_runtime(
    config: RuntimeConfig | None = None,
    *,
    capabilities: RuntimeCapabilities | None = None,
) -> ResolvedRuntime:
    """Resolve a portable runtime configuration to a device and backend.

    The resolver does not compile kernels or allocate large tensors. Backend
    support for a particular numerical operation is checked again by that
    operation, allowing an automatic runtime to use Triton where available
    and PyTorch elsewhere.
    """

    config = RuntimeConfig() if config is None else config
    capabilities = (
        RuntimeCapabilities.detect() if capabilities is None else capabilities
    )
    device = (
        _auto_device(capabilities)
        if str(config.device) == "auto"
        else torch.device(config.device)
    )
    if device.type == "cuda" and not capabilities.cuda_available:
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not capabilities.mps_available:
        raise RuntimeError("MPS was requested but is not available")
    if device.type == "mps" and config.dtype == torch.float64:
        raise RuntimeError("PyTorch MPS does not support the float64 package contract")

    requested = Backend(config.backend)
    reason: str | None = None
    compile_supported = _torch_compile_supported(device, capabilities)
    if requested is Backend.AUTO:
        if (
            device.type == "cuda"
            and config.dtype == torch.float32
            and capabilities.triton_importable
        ):
            backend = Backend.TRITON
        elif compile_supported:
            backend = Backend.TORCH_COMPILE
        else:
            backend = Backend.TORCH_EAGER
    elif requested is Backend.TRITON:
        supported = (
            device.type == "cuda"
            and config.dtype == torch.float32
            and capabilities.triton_importable
        )
        if supported:
            backend = requested
        elif config.strict_backend:
            raise RuntimeError(
                "Triton requires an importable Triton package and CUDA float32"
            )
        else:
            backend = (
                Backend.TORCH_COMPILE if compile_supported else Backend.TORCH_EAGER
            )
            reason = "Triton is unavailable for the selected device or dtype"
    elif requested is Backend.TORCH_COMPILE:
        if compile_supported:
            backend = requested
        elif config.strict_backend:
            raise RuntimeError(
                "torch.compile is unavailable for the selected device or its "
                "required compiler toolchain could not be found"
            )
        else:
            backend = Backend.TORCH_EAGER
            reason = (
                "torch.compile is unavailable or its device toolchain "
                "could not be found"
            )
    else:
        backend = Backend.TORCH_EAGER

    return ResolvedRuntime(
        device=device,
        backend=backend,
        dtype=config.dtype,
        memory_fraction=float(config.memory_fraction),
        strict_backend=bool(config.strict_backend),
        torch_compile_mode=config.torch_compile_mode,
        warn_on_compile=bool(config.warn_on_compile),
        profiling=config.profiling,
        capabilities=capabilities,
        fallback_reason=reason,
    )
