#!/usr/bin/env python3
"""Post-install patches for the tvm-ffi Windows JIT. Run once after `pip install`:

    python scripts/patch_deps_windows.py

Two independent problems, both MSVC-only. Each patch is idempotent, and the script is a
no-op off Windows.

1. **nvcc host flags.** tvm-ffi's Windows nvcc flags are emitted as
   `-Xcompiler /std:c++17 /O2` (three tokens), so nvcc treats `/O2` as a stray input file
   ("A single input file is required..."), and the host compiles as C++17 while
   FreeToken's JIT kernels require C++20. Rewritten to a comma-joined
   `-Xcompiler /std:c++20,/O2` plus `/std:c++20`.

2. **`FunctionInfo` does not accept a reference-to-function-pointer.**
   `TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch, (&Kernel<...>::run))` expands to
   `FunctionInfo<decltype(Function)>` where `Function` is the *parenthesized* expression
   `(&Kernel<...>::run)`. MSVC deduces that as `R (*&)(Args...)` -- a reference to a
   function pointer -- rather than the plain `R (*)(Args...)` that GCC and Clang produce.
   tvm-ffi specialises `R(Args...)`, `R(*)(Args...)` and `R(&)(Args...)` but not
   `R(*&)(Args...)`, so the primary template is selected, `T::operator()` is looked up on a
   function pointer, and the JIT dies with:

       function_details.h(124): error C2825: 'T': must be a class or namespace ...
       cuda.cu(8): error C2039: 'RetType': is not a member of ... FunctionInfo<void (__cdecl *&)(...)>

   This surfaces only on kernels compiled through nvcc (the MoE **offload** path, for
   instance) -- the fused path's Triton kernels never touch this header, which is why a
   fused model serves fine and an offloaded one fails minutes into startup.

   We add the two missing specialisations rather than editing the generated source,
   because the generated source is rewritten on every JIT.
"""
from __future__ import annotations

import sys
from pathlib import Path

_FLAG_REPLACEMENTS = [
    (
        'default_cuda_cflags = ["-Xcompiler", "/std:c++17", "/O2"]',
        'default_cuda_cflags = ["-Xcompiler", "/std:c++20,/O2"]',
    ),
    (
        'default_cxxflags = ["/std:c++17", "/MD", "/EHsc"]',
        'default_cxxflags = ["/std:c++20", "/MD", "/EHsc"]',
    ),
]

# Anchor: the last of tvm-ffi's own plain-function specialisations. We append after it so
# the new ones sit with their siblings and ahead of the member-pointer overloads.
_ANCHOR = (
    "template <typename R, typename... Args>\n"
    "struct FunctionInfo<R (&)(Args...), void> : FuncFunctorImpl<R, Args...> {};\n"
)

_ADDITION = """
// --- FreeToken Windows patch -------------------------------------------------------
// MSVC deduces `decltype((&Klass::static_method))` as a REFERENCE to a function pointer,
// so TVM_FFI_DLL_EXPORT_TYPED_FUNC instantiates FunctionInfo<R (*&)(Args...)>. Without
// these two specialisations that falls through to the primary template, which tries
// `decltype(&T::operator())` on a function pointer and fails to compile.
template <typename R, typename... Args>
struct FunctionInfo<R (*&)(Args...), void> : FuncFunctorImpl<R, Args...> {};
template <typename R, typename... Args>
struct FunctionInfo<R (*const&)(Args...), void> : FuncFunctorImpl<R, Args...> {};
// --- end FreeToken Windows patch ---------------------------------------------------
"""

_MARKER = "FreeToken Windows patch"


def _patch_flags() -> int:
    import tvm_ffi.cpp.extension as ext

    path = Path(ext.__file__)
    text = path.read_text(encoding="utf-8")
    changed = 0
    for old, new in _FLAG_REPLACEMENTS:
        if old in text:
            text = text.replace(old, new)
            changed += 1
    if changed:
        path.write_text(text, encoding="utf-8")
    print(f"  nvcc host flags: {changed} change(s) in {path.name}")
    return changed


def _patch_function_info() -> int:
    import tvm_ffi

    header = Path(tvm_ffi.__file__).parent / "include" / "tvm" / "ffi" / "function_details.h"
    if not header.is_file():
        print(f"  FunctionInfo: header not found at {header} -- skipped")
        return 0

    text = header.read_text(encoding="utf-8")
    if _MARKER in text:
        print("  FunctionInfo: already patched")
        return 0
    if _ANCHOR not in text:
        # Fail loudly rather than silently leaving the JIT broken: a tvm-ffi release that
        # moved this code needs a human to re-check the specialisation list.
        print(
            "  FunctionInfo: anchor not found -- tvm-ffi may have changed.\n"
            f"    Inspect {header} and add specialisations for R (*&)(Args...).",
            file=sys.stderr,
        )
        return 0

    header.write_text(text.replace(_ANCHOR, _ANCHOR + _ADDITION, 1), encoding="utf-8")
    print(f"  FunctionInfo: added R (*&)(Args...) specialisations in {header.name}")
    return 1


def main() -> int:
    if sys.platform != "win32":
        print("non-Windows: nothing to patch")
        return 0
    print("patching tvm-ffi for Windows:")
    _patch_flags()
    _patch_function_info()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
