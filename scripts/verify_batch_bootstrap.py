"""Exercise the actual BAT installers in a separate, disposable source copy.

Requires Windows, Python and package-index access. Leaves evidence under build/.
Never modifies the developer's .venv or opens a GUI.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    # Keep enough space for Qt's long wheel paths on Windows without long-path
    # support, while still exercising a source folder containing a space.
    run = root / 'build' / ('bat ' + time.strftime('%H%M%S'))
    run.mkdir(parents=True, exist_ok=False)
    assert run.resolve().is_relative_to((root / 'build').resolve())
    sources = subprocess.check_output(['git', 'ls-files', 'src'], cwd=root, text=True).splitlines()
    sources += ['AngleCal.bat', 'RegionLab.bat', 'setup_angle_cal.bat',
                'run_angle_cal.py', 'run_region_lab.py', 'pyproject.toml', 'README.md']
    for name in sources:
        target = run / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.bat'):
            target.write_bytes((root / name).read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
        else:
            shutil.copy2(root / name, target)
    env = dict(os.environ, ANGLECAL_NO_PAUSE='1')
    env.pop('PYTHONPATH', None)
    started = time.monotonic()

    def command(args, extra_env=None):
        print('RUN', args, flush=True)
        subprocess.run(args, cwd=run, env=dict(env, **(extra_env or {})), check=True, timeout=600)

    def bat(name, *args, offline=False):
        command([os.environ.get('COMSPEC', 'cmd.exe'), '/d', '/c', name, *args],
                {'PIP_NO_INDEX': '1'} if offline else None)

    # No .venv and no ready marker: first launch installs all dependencies.
    bat('RegionLab.bat', '--self-test', 'first-run.json')
    assert json.loads((run / 'first-run.json').read_text())['ok']
    # No index access on a healthy second launch.
    bat('AngleCal.bat', '--setup-only', offline=True)
    bat('RegionLab.bat', '--self-test', 'offline-run.json', offline=True)
    assert json.loads((run / 'offline-run.json').read_text())['ok']
    # Simulate a legacy installation whose old readiness marker hides missing skimage.
    python = run / '.venv' / 'Scripts' / 'python.exe'
    command([str(python), '-m', 'pip', 'uninstall', '-y', 'scikit-image'])
    (run / '.venv' / '.anglecal-ready').write_text('ok')
    bat('AngleCal.bat', '--setup-only')
    bat('RegionLab.bat', '--self-test', 'repaired-run.json', offline=True)
    assert json.loads((run / 'repaired-run.json').read_text())['ok']
    summary = dict(ok=True, clean_install=True, offline_restart=True,
                   legacy_missing_package_repaired=True, seconds=time.monotonic()-started,
                   runner=str(run))
    (run / 'bootstrap-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
