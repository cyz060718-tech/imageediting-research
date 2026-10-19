"""Load the Windows system VC runtime before old Conda DLLs are selected by torch."""
import os
_handles=[]
if os.name=='nt':
    import ctypes
    from pathlib import Path
    for name in ['vcruntime140.dll','vcruntime140_1.dll','msvcp140.dll']:
        path=Path(os.environ.get('SystemRoot','C:/Windows'))/'System32'/name
        if path.exists():
            _handles.append(ctypes.WinDLL(str(path)))
