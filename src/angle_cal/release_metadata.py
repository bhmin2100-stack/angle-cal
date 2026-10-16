"""Build identity embedded in each EXE without changing tracked source files."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def embedded_metadata(path=None):
    try:
        data=json.loads(Path(path or Path(__file__).with_name('_build_info.json')).read_text(encoding='utf-8-sig'))
        keys=('app_version','build_id','build_commit','build_date','update_channel')
        if not isinstance(data,dict) or any(not isinstance(data.get(k),str) or not data[k].strip() for k in keys):
            return {}
        if data['update_channel'] not in ('personal','company'):
            return {}
        return {k:data[k] for k in keys}
    except (OSError,ValueError,TypeError):
        return {}


def make_build_metadata(root, defaults, env=None, now=None):
    env=os.environ if env is None else env
    now=now or datetime.now(timezone.utc)
    stamp=now.astimezone(timezone.utc)
    commit=env.get('GITHUB_SHA') or defaults.get('build_commit','local')
    if commit in ('','local'):
        try:
            commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError):
            commit='local'
    build_id=defaults.get('build_id','local')
    if build_id in ('','local'):
        run=env.get('GITHUB_RUN_NUMBER')
        build_id=(f"{stamp:%Y%m%d}.{run}.{env.get('GITHUB_RUN_ATTEMPT','1')}"
                  if run else f"{stamp:%Y%m%d.%H%M%S}")
    return dict(app_version=defaults['app_version'],build_id=build_id,build_commit=commit,
                build_date=defaults.get('build_date') or stamp.isoformat(),
                update_channel=defaults.get('update_channel','personal'))


def release_manifest(exe, metadata, download_url):
    exe=Path(exe)
    digest=hashlib.sha256()
    with exe.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return dict(version=metadata['app_version'],asset=exe.name,downloadUrl=download_url,
                build_id=metadata['build_id'],commit=metadata['build_commit'],
                publishedAtUtc=metadata['build_date'],sha256=digest.hexdigest(),size=exe.stat().st_size)
