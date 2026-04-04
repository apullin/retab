"""Patch ExportPassBase.call_module to skip _guards_fn nodes."""
import pathlib
p = pathlib.Path("/work/ws/executorch/src/executorch/exir/pass_base.py")
if not p.exists():
    exit(0)
src = p.read_text()
old = 'raise ExportPassBaseError("call_module is not supported.")'
new = 'if "_guards" in str(target): return None\n            raise ExportPassBaseError("call_module is not supported.")'
if old in src and '"_guards"' not in src:
    p.write_text(src.replace(old, new))
    print("[docker] Patched pass_base.py to skip _guards_fn")
