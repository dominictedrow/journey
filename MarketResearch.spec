# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Market Research desktop app.

Bundles desktop.py (the pywebview entry point) plus the templates, static
assets, sources.json, and schema.sql that Flask/scheduler/init read at runtime
through config.settings.resource_path() (which honors sys._MEIPASS when frozen).

Build via:  bash scripts/build_app.sh   (generates build/AppIcon.icns first)
or directly: python3 -m PyInstaller MarketResearch.spec --noconfirm --clean
"""

# (source on disk, destination dir inside the bundle) — must mirror the
# project-relative paths passed to resource_path() so lookups resolve.
datas = [
    ('app/templates', 'app/templates'),
    ('app/static', 'app/static'),
    ('config/sources.json', 'config'),
    ('db/schema.sql', 'db'),
]

# psycopg2 and webview are imported lazily (inside functions), and the
# collectors are pulled in by the scheduler — name them explicitly so the
# frozen bundle includes them.
hiddenimports = [
    'psycopg2',
    'webview',
    'collectors.fred',
    'collectors.bls',
    'collectors.prices',
    'collectors.crypto',
    'collectors.etf_flows',
    'collectors.onchain',
    'collectors.news',
    'collectors.imf',
    'collectors.cftc',
    'collectors.holdings',
]

a = Analysis(
    ['desktop.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
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
    [],
    exclude_binaries=True,
    name='Market Research',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='build/AppIcon.icns',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Market Research',
)

app = BUNDLE(
    coll,
    name='Market Research.app',
    icon='build/AppIcon.icns',
    bundle_identifier='com.newsbot.marketresearch',
    info_plist={
        'NSHighResolutionCapable': True,
        'LSBackgroundOnly': False,
        'CFBundleName': 'Market Research',
        'CFBundleDisplayName': 'Market Research',
    },
)
