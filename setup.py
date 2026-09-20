from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension


ROOT = Path(__file__).parent
_WIN = sys.platform == "win32"
# MSVC uses /O2 and /std:c++17; POSIX threads are built-in on Windows (no -pthread).
_CXX = ["/O2", "/std:c++17", "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN"] if _WIN else ["-O3", "-std=c++17"]
_CXX_MT = _CXX if _WIN else _CXX + ["-pthread"]


def _check_toolchain() -> None:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _cuda_runtime_paths() -> tuple[list[str], list[str]]:
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA runtime API."
        )
    cuda_home = Path(CUDA_HOME)
    library_dirs = [str(cuda_home / "lib64")]
    if (cuda_home / "lib").exists():
        library_dirs.append(str(cuda_home / "lib"))
    if (cuda_home / "lib" / "x64").exists():  # Windows CUDA toolkit layout
        library_dirs.append(str(cuda_home / "lib" / "x64"))
    return [str(cuda_home / "include")], library_dirs


cuda_include_dirs, cuda_library_dirs = _cuda_runtime_paths()
_check_toolchain()


setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=_CXX,
        ),
        CppExtension(
            name="freetoken.kernel._cpu_moe",
            sources=[
                "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=_CXX_MT,
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
