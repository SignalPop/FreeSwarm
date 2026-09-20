"""Cache-bypassing sequential file reads on Windows -- the O_DIRECT analog.

The checkpoint readers take an ``O_DIRECT`` path on Linux precisely so that reading tens of
GiB of weights does not also push tens of GiB through the page cache. Their Windows
fallbacks used ordinary buffered ``readinto``, which means a 72 GiB checkpoint load also
materialises ~72 GiB of file cache -- on top of the page-locked expert banks, which are not
reclaimable. That combination is what leaves the machine unresponsive during a load.

``CreateFileW`` with ``FILE_FLAG_NO_BUFFERING`` gives the same guarantee ``O_DIRECT`` does:
the DMA lands in the caller's buffer and nothing is retained by the cache manager. Its
constraint is alignment -- buffer address, file offset and read length must all be multiples
of the volume's sector size. Both callers read into page-aligned ``mmap`` buffers rounded up
to 4096 in 8 MiB chunks, so the only special case is the final chunk, where a sector-multiple
request legitimately returns short at EOF.

Anything that cannot be verified (unusual sector size, a handle that will not open, a buffer
that is not aligned) reports failure and the caller keeps its buffered path: this module only
ever makes a load use *less* memory, never fail.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import sys

_BLK = 4096  # alignment the callers already guarantee

_GENERIC_READ = 0x80000000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_OPEN_EXISTING = 3
_FILE_FLAG_NO_BUFFERING = 0x20000000
_FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


def available() -> bool:
    """True if unbuffered reads can be attempted on this platform."""
    return sys.platform == "win32"


def _sector_size(path: str) -> int | None:
    """Bytes per sector of the volume holding ``path``, or ``None`` if unknown."""
    try:
        root = os.path.splitdrive(os.path.abspath(path))[0]
        if not root:
            return None
        sectors_per_cluster = ctypes.c_ulong()
        bytes_per_sector = ctypes.c_ulong()
        free_clusters = ctypes.c_ulong()
        total_clusters = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetDiskFreeSpaceW(
            ctypes.c_wchar_p(root + "\\"),
            ctypes.byref(sectors_per_cluster),
            ctypes.byref(bytes_per_sector),
            ctypes.byref(free_clusters),
            ctypes.byref(total_clusters),
        )
        if not ok or bytes_per_sector.value == 0:
            return None
        return int(bytes_per_sector.value)
    except (OSError, AttributeError, ValueError):
        return None


def _buffer_address(buf) -> int | None:
    """Base address of a writable buffer, or ``None`` if it cannot be pinned down."""
    try:
        return ctypes.addressof(ctypes.c_char.from_buffer(buf))
    except (TypeError, BufferError, ValueError):
        return None


def read_into(path: str, buf, *, chunk: int = 8 << 20) -> int | None:
    """Read all of ``path`` into ``buf`` bypassing the system file cache.

    ``buf`` is an ``mmap`` or writable ``memoryview`` whose base address is 4096-aligned and
    whose length covers the file size rounded up to 4096. Returns the file size on success,
    or ``None`` if unbuffered I/O is unusable here -- the caller must then fall back to a
    buffered read.
    """
    if sys.platform != "win32":
        return None

    sector = _sector_size(path)
    if sector is None or _BLK % sector != 0:
        return None  # sector > 4096 (or unknown): our 4096 alignment is not sufficient

    base = _buffer_address(buf)
    if base is None or base % _BLK != 0:
        return None

    size = os.path.getsize(path)
    capacity = len(buf)
    if capacity < ((size + _BLK - 1) // _BLK) * _BLK:
        return None  # caller's buffer cannot hold the sector-rounded tail read

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateFileW.restype = ctypes.c_void_p
    handle = kernel32.CreateFileW(
        ctypes.c_wchar_p(path),
        ctypes.c_uint(_GENERIC_READ),
        ctypes.c_uint(_FILE_SHARE_READ | _FILE_SHARE_WRITE),
        None,
        ctypes.c_uint(_OPEN_EXISTING),
        ctypes.c_uint(_FILE_FLAG_NO_BUFFERING | _FILE_FLAG_SEQUENTIAL_SCAN),
        None,
    )
    if handle is None or handle == _INVALID_HANDLE_VALUE:
        return None

    # Chunk must stay sector-aligned or every read after the first is misaligned.
    chunk = max(_BLK, (chunk // _BLK) * _BLK)
    read = ctypes.c_ulong()
    off = 0
    try:
        while off < size:
            # Round the request up to a whole sector: NO_BUFFERING forbids a partial-sector
            # length, and ReadFile is documented to return short at EOF, which is the tail.
            want = min(chunk, ((size - off + _BLK - 1) // _BLK) * _BLK)
            ok = kernel32.ReadFile(
                ctypes.c_void_p(handle),
                ctypes.c_void_p(base + off),
                ctypes.c_ulong(want),
                ctypes.byref(read),
                None,
            )
            if not ok:
                raise OSError(ctypes.get_last_error(), f"ReadFile failed at offset {off}")
            if read.value == 0:
                break  # EOF earlier than getsize claimed; caller sees the short return
            off += read.value
    except OSError:
        return None
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
    return size if off >= size else None


def discard(buf: mmap.mmap) -> bool:
    """Drop the resident pages of an anonymous ``mmap`` (the ``MADV_DONTNEED`` analog).

    ``DiscardVirtualMemory`` keeps the reservation valid and makes the contents undefined,
    which is exactly the contract of the release paths that call it. Returns False when the
    call is unavailable (pre-Windows 8) or fails, so callers can treat it as best-effort.
    """
    if sys.platform != "win32":
        return False
    base = _buffer_address(buf)
    if base is None:
        return False
    try:
        discard_fn = ctypes.windll.kernel32.DiscardVirtualMemory
    except AttributeError:
        return False
    try:
        # Returns ERROR_SUCCESS (0) on success, unlike most kernel32 BOOL functions.
        return discard_fn(ctypes.c_void_p(base), ctypes.c_size_t(len(buf))) == 0
    except OSError:
        return False


__all__ = ["available", "discard", "read_into"]
