import os
import hashlib
import threading
from pathlib import Path
from collections import OrderedDict

import torch

class TwoTierFeatureCache:
    """
    LRU cache with a RAM pool (byte-capped) and a disk pool (byte-capped).
    Values are dictionaries of tensors; we store CPU copies on write and
    restore to a requested device on read, **preserving original dtype**.

    Notes:
      - Thread-safe for simple single-process use (threading.Lock).
      - We write .pt files atomically and maintain an in-memory LRU index.
      - For multi-process robustness, prefer a library like `diskcache`.
    """
    def __init__(self, mem_limit_bytes=16 * (1024**3),   # 32GB default
                 disk_limit_bytes=None,     # None = unlimited
                 cache_dir=None,
                 cache_hidden_states=False,       # hidden_states can be HUGE
    ):
        self.mem_limit = int(mem_limit_bytes)
        self.disk_limit = None if disk_limit_bytes is None else int(disk_limit_bytes)
        self.cache_hidden_states = bool(cache_hidden_states)

        if cache_dir is None:
            base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            cache_dir = base / "dinov3_feature_cache"
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        # RAM store: key -> value (CPU tensors), LRU via OrderedDict
        self._ram = OrderedDict()
        self._ram_bytes = 0

        # Disk index: key -> {"file": filename, "bytes": size}
        self._disk = OrderedDict()
        self._disk_bytes = 0

        self._lock = threading.Lock()

    # ---------- helpers ----------
    def _sizeof(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        if isinstance(obj, dict):
            return sum(self._sizeof(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(self._sizeof(v) for v in obj)
        return 0

    def _to_cpu_detached(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().to("cpu", copy=True)
        if isinstance(obj, dict):
            return {k: self._to_cpu_detached(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(self._to_cpu_detached(v) for v in obj)
        return obj

    def _to_device(self, obj, device):
        """Move to device, preserving dtype."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device=device, non_blocking=False)
        if isinstance(obj, dict):
            return {k: self._to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(self._to_device(v, device) for v in obj)
        return obj

    def _key_to_filename(self, key):
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return f"{digest}.pt"

    def _atomic_save(self, data, path):
        tmp = path.with_suffix(".pt.tmp")
        torch.save(data, tmp)
        os.replace(tmp, path)

    def _evict_ram_until(self, needed_bytes):
        # Move oldest RAM items to disk until enough room for needed_bytes
        while self._ram and (self._ram_bytes + needed_bytes > self.mem_limit):
            k, v = self._ram.popitem(last=False)
            b = self._sizeof(v)
            # ensure disk space or drop oldest disk items
            self._ensure_disk_space(b)
            filename = self._key_to_filename(k)
            fpath = self.dir / filename
            self._atomic_save(v, fpath)
            self._disk[k] = {"file": filename, "bytes": b}
            self._disk_bytes += b
            self._ram_bytes -= b

    def _ensure_disk_space(self, needed_bytes):
        if self.disk_limit is None:
            return
        while self._disk and (self._disk_bytes + needed_bytes > self.disk_limit):
            dk, meta = self._disk.popitem(last=False)  # oldest on disk
            try:
                os.remove(self.dir / meta["file"])
            except FileNotFoundError:
                pass
            self._disk_bytes -= meta["bytes"]

    # ---------- public API ----------
    def get(self, key, device):
        """
        Return cached value moved to `device`, preserving original tensor dtypes.
        """
        with self._lock:
            # RAM hit
            if key in self._ram:
                val = self._ram.pop(key)
                self._ram[key] = val  # move to MRU
                return self._to_device(val, device)

            # Disk hit -> promote to RAM (and remove file)
            if key in self._disk:
                meta = self._disk.pop(key)
                fpath = self.dir / meta["file"]
                try:
                    cpu_val = torch.load(fpath, map_location="cpu", weights_only=False)
                except Exception:
                    # Corrupt file or race; forget it
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass
                    return None

                # free disk space accounting
                try:
                    os.remove(fpath)
                except OSError:
                    pass
                self._disk_bytes -= int(meta["bytes"])

                size = self._sizeof(cpu_val)
                self._evict_ram_until(size)
                self._ram[key] = cpu_val
                self._ram_bytes += size
                return self._to_device(cpu_val, device)

            return None

    def put(self, key, value):
        # drop hidden_states unless explicitly requested
        if not self.cache_hidden_states and isinstance(value, dict) and "hidden_states" in value:
            value = {k: v for k, v in value.items() if k != "hidden_states"}

        cpu_val = self._to_cpu_detached(value)
        size = self._sizeof(cpu_val)

        with self._lock:
            # If single item is bigger than the entire RAM budget, write straight to disk
            if size > self.mem_limit:
                self._ensure_disk_space(size)
                filename = self._key_to_filename(key)
                fpath = self.dir / filename
                self._atomic_save(cpu_val, fpath)
                self._disk[key] = {"file": filename, "bytes": size}
                self._disk_bytes += size
                return "disk"

            # Otherwise, make room in RAM and add
            self._evict_ram_until(size)
            self._ram[key] = cpu_val
            self._ram_bytes += size
            return "ram"

    def stats(self):
        with self._lock:
            return {
                "ram_items": len(self._ram),
                "ram_bytes": self._ram_bytes,
                "disk_items": len(self._disk),
                "disk_bytes": self._disk_bytes,
                "mem_limit": self.mem_limit,
                "disk_limit": self.disk_limit,
                "dir": str(self.dir),
            }

    def clear(self):
        with self._lock:
            self._ram.clear()
            self._ram_bytes = 0
            for _, meta in self._disk.items():
                try:
                    os.remove(self.dir / meta["file"])
                except OSError:
                    pass
            self._disk.clear()
            self._disk_bytes = 0
