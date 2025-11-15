# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import sys
import os

# ---------------------------------------------------------------------------
# Windows-only PyTorch workaround
# ---------------------------------------------------------------------------
# The default PyInstaller build flow performs many hook operations inside
# isolated child Python processes. Importing torch inside those workers causes
# an immediate crash on GitHub's Windows runners when torch tries to load its
# DLL stack (see `_load_dll_libraries`). PyInstaller 6.16 does not yet expose a
# `--no-isolate` switch, so we inline that behaviour here by monkey-patching the
# isolation helpers before Analysis starts. This keeps macOS/Linux builds
# unchanged while letting Windows finish Analysis without firing up child
# interpreters.
if sys.platform == 'win32':
    from PyInstaller import isolated as _isolated
    from PyInstaller.isolated import _parent as _isolated_parent

    def _noisolate_call(function, *args, **kwargs):
        return function(*args, **kwargs)

    # Existing decorated hook helpers capture the original function object, so
    # mutate it in-place to avoid touching every decorator.
    _isolated_parent.call.__code__ = _noisolate_call.__code__
    _isolated_parent.call.__defaults__ = _noisolate_call.__defaults__
    _isolated_parent.call.__kwdefaults__ = _noisolate_call.__kwdefaults__

    class _NoIsolatePython:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def call(self, function, *args, **kwargs):
            return function(*args, **kwargs)

    _isolated_parent.Python = _NoIsolatePython
    _isolated.Python = _NoIsolatePython
    _isolated.call = _isolated_parent.call

    def _noisolate_decorate(function):
        return function

    _isolated.decorate = _noisolate_decorate

# Get SarcAsM version
try:
    from sarcasm import __version__ as version
except ImportError:
    version = '0.0.0-import-error'

# Dynamic platform-aware naming
appname = f"SarcAsM-v{version}"

# Collect data files - be comprehensive, not selective
napari_data = collect_data_files('napari', include_py_files=False)
napari_data += collect_data_files('napari_builtins', include_py_files=False)
vispy_data = collect_data_files('vispy', include_py_files=False)
model_data = [('sarcasm/models', 'sarcasm/models')]

# Collect data for missing dependencies
rfc3987_syntax_data = collect_data_files('rfc3987_syntax')  # macOS: missing .lark grammar files
numpy_data = collect_data_files('numpy')  # Windows: ensure all DLLs/libs are collected

# Only exclude things that are definitely NOT needed
excludes = [
    # Exclude test frameworks and dev tools (but NOT unittest - torch needs it!)
    'pytest', '_pytest', 'nose', 'hypothesis',
    'IPython', 'jupyter', 'notebook', 'nbconvert', 'nbformat',
    'tkinter', 'tcl', 'tk', '_tkinter',
    # Exclude test submodules (but NOT numpy.testing - scipy needs it!)
    'scipy.tests',
    'matplotlib.tests',
    'PIL.ImageQt',
]

# Platform-specific excludes for Windows to avoid torch import issues during analysis
if sys.platform == 'win32':
    # Don't let PyInstaller try to import torch during analysis - causes access violations
    # The hook will collect it properly instead
    pass  # We'll handle torch via hook, not via excludes

a = Analysis(
    ['sarcasm_app/__main__.py'],
    pathex=['.'],
    binaries=[],
    datas=napari_data + vispy_data + model_data + rfc3987_syntax_data + numpy_data,
    hiddenimports=[
        'napari',
        'napari._qt',
        'napari.plugins',
        'napari_builtins',
        'vispy',
        'vispy.glsl',
        'vispy.app',
        'vispy.app.backends',
        'vispy.app.backends._pyqt5',
        'PyQt5.QtOpenGL',
        'freetype',
        'PyQt5.sip',
        # Fix Windows numpy DLL issue
        'numpy.core._multiarray_umath',
        'numpy._core._multiarray_umath',
        # Fix macOS rfc3987 missing module
        'rfc3987',
        'rfc3987_syntax',
        'rfc3987_syntax.syntax_helpers',
    ] + collect_submodules('sarcasm_app') + collect_submodules('vispy'),
    hookspath=['sarcasm_app/hooks'],
    hooksconfig={},
    runtime_hooks=['sarcasm_app/hooks/runtime_hook_matplotlib.py'],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=appname,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Faster startup without compression
    upx_exclude=[],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='sarcasm_app/icons/sarcasm.ico',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name=f'{appname}.app',
        icon='sarcasm_app/icons/sarcasm.icns',
        bundle_identifier='de.umg.sarcasm',
        info_plist={
            'CFBundleName': 'SarcAsM',
            'CFBundleDisplayName': 'SarcAsM',
            'CFBundleShortVersionString': version,
            'CFBundleVersion': version,
            'NSHighResolutionCapable': 'True',
            'LSUIElement': 'False',
            'NSRequiresAquaSystemAppearance': 'False',
        }
    )
else:
    pass
