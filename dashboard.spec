# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for HolonicTrader Dashboard

block_cipher = None

a = Analysis(
    ['dashboard_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('market_data', 'market_data'),  # Include market data folder
        ('holonic_trader.db', '.'),      # Include database
        ('config.py', '.'),               # Include config
    ],
    hiddenimports=[
        'tkinter',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'numpy',
        'pandas',
        'ccxt',
        'tensorflow',
        'keras',
        'sqlite3',
        'queue',
        'threading',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HolonicTrader',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon later if desired
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HolonicTrader',
)
