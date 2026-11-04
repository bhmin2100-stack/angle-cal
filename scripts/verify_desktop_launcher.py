"""Check Windows BAT downloads with no Python on PATH and a separate cache.

Requires access to the public pinned GitHub release assets. Leaves reports in
build/ and never changes the user's runtime cache or Python environment.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    if os.name != 'nt':
        raise SystemExit('This integration check requires Windows')
    root = Path(__file__).resolve().parents[1]
    run = root / 'build' / ('desktop ' + time.strftime('%Y%m%d-%H%M%S'))
    run.mkdir(parents=True, exist_ok=False)
    for name in ('AngleCal.bat', 'RegionLab.bat', 'launch_desktop.bat', 'desktop_runtime.bat'):
        content = (root / name).read_bytes().replace(b'\r\n', b'\n')
        (run / name).write_bytes(content.replace(b'\n', b'\r\n'))
    system = Path(os.environ['SystemRoot'])
    env = dict(os.environ, ANGLECAL_NO_PAUSE='1', LOCALAPPDATA=str(run / 'cache'))
    env['PATH'] = os.pathsep.join(map(str, (system / 'System32',
                                          system / 'System32/WindowsPowerShell/v1.0')))
    for executable in ('python', 'python3', 'pip', 'py'):
        assert shutil.which(executable, path=env['PATH']) is None, executable
    env.pop('PYTHONPATH', None)
    env.pop('VIRTUAL_ENV', None)
    offline = dict(env, HTTPS_PROXY='http://127.0.0.1:9', HTTP_PROXY='http://127.0.0.1:9',
                   ALL_PROXY='http://127.0.0.1:9', NO_PROXY='')

    def launch(app, report, *, network=True, success=True):
        command = [str(system / 'System32/cmd.exe'), '/d', '/c',
                   app + '.bat', '--verify', report + '.json']
        result = subprocess.run(command, cwd=run, env=env if network else offline,
                                capture_output=True, timeout=300)
        (run / (report + '.log')).write_bytes(result.stdout + result.stderr)
        print(app, report, 'exit', result.returncode, flush=True)
        if not success:
            assert result.returncode != 0, result.stdout
            assert not (run / (report + '.json')).exists()
            return
        assert result.returncode == 0, (result.stdout, result.stderr)
        data = json.loads((run / (report + '.json')).read_text(encoding='utf-8'))
        assert data.get('ok') if app == 'RegionLab' else data.get('build_commit')

    for app in ('AngleCal', 'RegionLab'):
        launch(app, app + '-first')
        launch(app, app + '-offline', network=False)
    # A changed cached EXE must never execute; fail offline, then repair online.
    cached = list((run / 'cache').rglob('RegionLab.exe'))
    assert len(cached) == 1
    with cached[0].open('ab') as stream:
        stream.write(b'launcher validation: modified cache')
    launch('RegionLab', 'modified-cache-rejected', network=False, success=False)
    launch('RegionLab', 'modified-cache-repaired')
    assert not (run / '.venv').exists()
    summary = dict(ok=True, no_python_on_path=True, fresh_downloads=True,
                   cached_offline_launch=True, modified_exe_rejected=True,
                   modified_exe_repaired=True, evidence=str(run))
    (run / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
