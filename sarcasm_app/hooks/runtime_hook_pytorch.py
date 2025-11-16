"""
Runtime hook for PyTorch - adds torch/lib to DLL search path on Windows.
This MUST run before torch is imported.
"""
import sys
import os

if sys.platform == 'win32':
    # When running as PyInstaller bundle
    if getattr(sys, 'frozen', False):
        import_path = getattr(sys, '_MEIPASS', None)
        if import_path:
            # Add torch/lib to DLL search path
            torch_lib = os.path.join(import_path, 'torch', 'lib')
            if os.path.exists(torch_lib):
                # Windows 10+ method
                try:
                    os.add_dll_directory(torch_lib)
                    print(f"[Runtime Hook] Added torch/lib to DLL search: {torch_lib}")
                except (OSError, AttributeError):
                    # Windows 7 fallback
                    os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
                    print(f"[Runtime Hook] Added torch/lib to PATH: {torch_lib}")
