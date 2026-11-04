# Standalone experiment build: does not replace the installed AngleCal EXE.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

a = Analysis(['run_region_lab.py'], pathex=['src'],
             datas=collect_data_files('skimage'),
             hiddenimports=['skimage.segmentation._felzenszwalb_cy']
             + collect_submodules('scipy._external.array_api_compat.numpy'),
             excludes=['matplotlib', 'tkinter', 'IPython'])
# Preserve the same Windows ICU isolation as AngleCal.spec.
foreign_icu_dirs = {
    Path(source).parent for dest, source, _ in a.binaries
    if Path(dest).name.lower() in {'icuuc.dll', 'icuin.dll', 'icu.dll'}
    and Path(source).parent.name.lower() != 'pyside6'
}
a.binaries = [entry for entry in a.binaries
              if not (Path(entry[1]).parent in foreign_icu_dirs
                      and Path(entry[0]).name.lower().startswith('icu')
                      and Path(entry[0]).suffix.lower() == '.dll')]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='RegionLab',
          console=False, icon='src/angle_cal/assets/anglecal_icon.ico')
coll = COLLECT(exe, a.binaries, a.datas, name='RegionLab')
