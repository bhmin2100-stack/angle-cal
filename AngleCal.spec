# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules
from pathlib import Path


hiddenimports = collect_submodules("cv2")

a = Analysis(
    ["run_angle_cal.py"],
    pathex=[".", "src"],
    binaries=[],
    datas=[("src/angle_cal/assets/anglecal_icon.png", "angle_cal/assets")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
# Qt uses the Windows-provided, unversioned ICU API. A Poppler/Conda directory
# on the build host's PATH can supply a different icuuc.dll whose exports are
# version-suffixed, making QtCore fail only in the frozen application. Keep the
# Windows ICU loader names unbundled, as with other operating-system DLLs, and
# discard the data DLL pulled in solely by that unrelated ICU implementation.
windows_icu_names = {"icuuc.dll", "icuin.dll", "icu.dll"}
foreign_icu_dirs = {
    Path(source).parent for dest, source, _ in a.binaries
    if Path(dest).name.lower() in windows_icu_names
    and Path(source).parent.name.lower() != "pyside6"
}
a.binaries = [
    entry for entry in a.binaries
    if not (Path(entry[1]).parent in foreign_icu_dirs
            and Path(entry[0]).name.lower().startswith("icu")
            and Path(entry[0]).suffix.lower() == ".dll")
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    exclude_binaries=False,
    name="AngleCal",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="src/angle_cal/assets/anglecal_icon.ico",
)
