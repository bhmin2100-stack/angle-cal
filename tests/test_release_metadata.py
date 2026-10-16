from datetime import datetime,timezone
import hashlib
import json
from unittest.mock import patch

from PySide6.QtWidgets import QApplication
from angle_cal import updater
from angle_cal.app import MainWindow
from angle_cal.release_metadata import embedded_metadata,make_build_metadata,release_manifest


def defaults(**extra):
    return dict(app_version='0.1.1',build_id='local',build_date='',build_commit='local',update_channel='personal')|extra


def test_github_build_and_attempt_get_distinct_automatic_versions(tmp_path):
    now=datetime(2026,10,16,8,30,tzinfo=timezone.utc)
    env=dict(GITHUB_RUN_NUMBER='42',GITHUB_RUN_ATTEMPT='1',GITHUB_SHA='a'*40)
    first=make_build_metadata(tmp_path,defaults(),env,now)
    second=make_build_metadata(tmp_path,defaults(),dict(env,GITHUB_RUN_ATTEMPT='2'),now)
    assert first['build_id']=='20261016.42.1'
    assert second['build_id']=='20261016.42.2'
    assert first['build_commit']=='a'*40
    assert first['build_date']=='2026-10-16T08:30:00+00:00'


def test_local_build_stamped_without_editing_company_identity(tmp_path):
    now=datetime(2026,10,16,8,30,15,tzinfo=timezone.utc)
    local=make_build_metadata(tmp_path,defaults(),{},now)
    assert local['build_id']=='20261016.083015'
    company=defaults(build_id='company-123',build_commit='b'*40,build_date='custom-date',update_channel='company')
    assert make_build_metadata(tmp_path,company,{},now)==company


def test_embedded_metadata_recovers_from_missing_and_malformed_file(tmp_path):
    path=tmp_path/'_build_info.json'
    assert embedded_metadata(path)=={}
    path.write_text('broken',encoding='utf-8');assert embedded_metadata(path)=={}
    path.write_text('{}',encoding='utf-8');assert embedded_metadata(path)=={}
    data=make_build_metadata(tmp_path,defaults(),{})
    path.write_text(json.dumps(data),encoding='utf-8')
    assert embedded_metadata(path)==data


def test_release_manifest_uses_same_identity_and_real_binary_hash(tmp_path):
    exe=tmp_path/'AngleCal.exe';exe.write_bytes(b'packaged app bytes')
    metadata=make_build_metadata(tmp_path,defaults(),{'GITHUB_RUN_NUMBER':'42'})
    url='https://example.com/AngleCal.exe'
    manifest=release_manifest(exe,metadata,url)
    assert manifest['sha256']==hashlib.sha256(exe.read_bytes()).hexdigest()
    assert manifest['size']==len(exe.read_bytes())
    assert manifest['build_id']==metadata['build_id']
    with patch.object(updater,'BUILD_ID','older-build'),patch.object(updater,'current_version',return_value=metadata['app_version']):
        info=updater._update_info_from_release(updater.PERSONAL_CHANNEL,{},manifest,None)
    assert info.is_available
    assert info.latest_label==f"{metadata['app_version']} ({metadata['build_id']})"


def test_real_window_title_uses_current_embedded_identity():
    app=QApplication.instance() or QApplication([])
    with patch.object(updater,'APP_VERSION','0.1.1'),patch.object(updater,'BUILD_ID','20261016.42.1'):
        window=MainWindow()
        assert window.windowTitle()=='AngleCal v0.1.1 · 빌드 20261016.42.1 - SEM Angle Measurement'
        window.close()
    with patch.object(updater,'BUILD_ID','local'):
        assert '개발용' in updater.display_version()
