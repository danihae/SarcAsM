# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import sys

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

# 3. Include your models directory (recursive)
model_data = [
    ('models/*', 'models'),  # Recursive inclusion
]

a = Analysis(
    ['sarcasm_app/__main__.py'],
    pathex=['.'],  # Project root
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
    a.binaries,
    a.datas,
    [],
    name=appname,  # Dynamic versioned name
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='docs/images/sarcasm.ico',
)

# Platform-specific configurations
if sys.platform == 'darwin':
    # macOS .app bundle configuration
    app = BUNDLE(
        exe,
        name=f'{appname}.app',  # .app extension for macOS
        icon='docs/images/sarcasm.icns',  # macOS requires .icns format
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
    # Windows/Linux configuration
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        name=appname,
    )
