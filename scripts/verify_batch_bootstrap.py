"""Exercise the actual BAT installers in a separate, disposable source copy.

Requires Windows, Python and package-index access. Leaves evidence under build/.
Never modifies the developer's .venv or opens a GUI.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    root = Path(__file__).resolve().parents[1]
    # Keep enough space for Qt's long wheel paths on Windows without long-path
    # support, while still exercising a source folder containing a space.
    run = root / 'build' / ('bat ' + time.strftime('%H%M%S'))
    run.mkdir(parents=True, exist_ok=False)
    assert run.resolve().is_relative_to((root / 'build').resolve())
    sources = subprocess.check_output(['git', 'ls-files', 'src'], cwd=root, text=True).splitlines()
    sources += ['AngleCal.bat', 'RegionLab.bat', 'AngleCal-source.bat', 'RegionLab-source.bat',
                'setup_angle_cal.bat', 'launch_source.bat', 'requirements.txt',
                'requirements-regions.txt', 'run_angle_cal.py', 'run_region_lab.py',
                'pyproject.toml', 'README.md']
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

    offline = dict(PIP_NO_INDEX='1', HTTP_PROXY='http://127.0.0.1:9',
                   HTTPS_PROXY='http://127.0.0.1:9', ALL_PROXY='http://127.0.0.1:9', NO_PROXY='')

    def command(args, extra_env=None, expected=0):
        print('RUN', args, flush=True)
        result = subprocess.run(args, cwd=run, env=dict(env, **(extra_env or {})),
                                capture_output=True, timeout=600)
        with (run / 'verification.log').open('ab') as log:
            log.write(result.stdout + result.stderr)
        assert result.returncode == expected, (result.returncode, result.stdout, result.stderr)
        return result

    def bat(name, *args, network=True, expected=0):
        return command([os.environ.get('COMSPEC', 'cmd.exe'), '/d', '/c', name, *args],
                       None if network else offline, expected)

    # Normal AngleCal installs only the dependencies that predate RegionLab.
    bat('AngleCal.bat', '--verify', 'base-first.json')
    python = run / '.venv' / 'Scripts' / 'python.exe'
    command([str(python), '-c', "import importlib.util; assert importlib.util.find_spec('skimage') is None"])
    pip_before = command([str(python), '-m', 'pip', '--version']).stdout
    (run / '.venv' / '.anglecal-ready').write_text('ok')

    # Source edits must take effect immediately, without editable installation.
    init = run / 'src/angle_cal/__init__.py'
    original_init = init.read_bytes()
    init.write_bytes(original_init + b"\n__version__ = '0.0.0-checkout-test'\n")
    bat('AngleCal.bat', '--verify', 'base-offline.json', network=False)
    assert json.loads((run / 'base-offline.json').read_text())['package_version'] == '0.0.0-checkout-test'
    init.write_bytes(original_init)
    assert b'Installing' not in (run / 'base-setup.log').read_bytes()

    # Failure to obtain optional packages cannot prevent the base app launching.
    failed = bat('RegionLab.bat', '--setup-only', network=False, expected=1)
    assert b'[AngleCal:REGION_DEPENDENCIES]' in failed.stdout
    bat('AngleCal-source.bat', '--verify', 'base-after-region-failure.json', network=False)
    bat('RegionLab.bat', '--self-test', 'region-first.json')
    assert json.loads((run / 'region-first.json').read_text())['ok']
    bat('RegionLab-source.bat', '--self-test', 'region-offline.json', network=False)
    assert json.loads((run / 'region-offline.json').read_text())['ok']
    assert command([str(python), '-m', 'pip', '--version']).stdout == pip_before

    # A startup exception must propagate through both BAT layers and leave a log.
    entry = run / 'run_angle_cal.py'
    original_entry = entry.read_bytes()
    entry.write_text("raise RuntimeError('startup-error-probe')\n", encoding='utf-8')
    try:
        failed = bat('AngleCal.bat', network=False, expected=1)
        assert b'[AngleCal:START]' in failed.stdout
        assert b'startup-error-probe' in (run / 'AngleCal-run.log').read_bytes()
    finally:
        entry.write_bytes(original_entry)
    summary = dict(ok=True, clean_base_install=True, base_without_regions_offline=True,
                   current_checkout_executed=True, optional_install_failure_isolated=True,
                   segmentation=True, offline_restart=True, pip_not_upgraded=True,
                   startup_error_logged=True, seconds=time.monotonic()-started,
                   runner=str(run))
    (run / 'bootstrap-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
