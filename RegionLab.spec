# Standalone experiment build: does not replace the installed AngleCal EXE.
from pathlib import Path
import json
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

root = Path(SPECPATH)
sys.path.insert(0, str(root / 'src'))
from angle_cal import build_info
from angle_cal.release_metadata import make_build_metadata
metadata = make_build_metadata(root, dict(app_version=build_info.APP_VERSION,
    build_id=build_info.BUILD_ID, build_commit=build_info.BUILD_COMMIT,
    build_date=build_info.BUILD_DATE, update_channel=build_info.UPDATE_CHANNEL))
metadata['distribution'] = 'onedir'
metadata_path = root / 'build' / 'region-lab' / '_build_info.json'
metadata_path.parent.mkdir(parents=True, exist_ok=True)
metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

a = Analysis(['run_region_lab.py'], pathex=['src'],
             datas=collect_data_files('skimage') + [(str(metadata_path), 'angle_cal')],
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
