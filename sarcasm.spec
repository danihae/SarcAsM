# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import sys
import os

# ---------------------------------------------------------------------------
# Windows-only PyTorch workaround for GitHub Actions
# ---------------------------------------------------------------------------
if sys.platform == 'win32':
    from PyInstaller import isolated as _isolated
    from PyInstaller.isolated import _parent as _isolated_parent

    def _noisolate_call(function, *args, **kwargs):
        return function(*args, **kwargs)

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

# 1. Include Napari resources
napari_data = collect_data_files('napari')

# 2. Include Vispy resources (critical for GLSL shaders)
vispy_data = collect_data_files('vispy')

# 3. Include models directory (recursive)
model_data = [
    ('sarcasm/models', 'sarcasm/models'),
]

a = Analysis(
    ['sarcasm_app/__main__.py'],
    pathex=['.'],
    binaries=[],
    datas=napari_data + vispy_data + model_data,
    hiddenimports=[
        'napari',
        'napari._qt',
        'napari.plugins',
        'vispy',
        'vispy.glsl',
        'vispy.app.backends._pyqt5',
        'freetype'
    ] + collect_submodules('sarcasm_app') + collect_submodules('vispy'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],                    # ← Empty - no binaries in exe
    exclude_binaries=True, # ← Critical for ONEDIR
    name=appname,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,             # ← Disabled for compatibility
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='sarcasm_app/icons/sarcasm.ico',
)

# Platform-specific configurations
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name=f'{appname}.app',
        icon='sarcasm_app/icons/sarcasm.icns',
        bundle_identifier='de.example.sarcasm',
        info_plist={
            'CFBundleName': 'SarcAsM',
            'CFBundleDisplayName': 'SarcAsM',
            'CFBundleShortVersionString': version,
            'CFBundleVersion': version,
            'NSHighResolutionCapable': 'True',
            'LSUIElement': 'False',
        }
    )
else:
    # Windows/Linux ONEDIR configuration
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        name=appname,
    )
