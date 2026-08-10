from __future__ import annotations

import tempfile
import tomllib
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import angle_cal
from angle_cal import updater


class UpdaterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.company = updater.COMPANY_CHANNEL
        self.release = {
            "tag_name": "0.1.1",
            "published_at": "2026-07-23T00:00:00Z",
            "body": "Release notes",
            "html_url": "https://github.samsungds.net/bh2-min/angle-cal/releases/1",
            "assets": [
                {"name": "AngleCal.exe", "browser_download_url": "https://company/AngleCal.exe", "size": 12},
                {"name": "version.json", "browser_download_url": "https://company/version.json"},
            ],
        }

    def test_company_release_manager_manifest_is_parsed(self) -> None:
        manifest = {
            "version": "0.1.1",
            "asset": "AngleCal.exe",
            "downloadUrl": "https://company/download/AngleCal.exe",
            "sha256": "abc",
            "notes": "Fixed angle copy",
            "publishedAtUtc": "2026-07-23T00:00:00Z",
        }

        info = updater._update_info_from_release(self.company, self.release, manifest, None)

        self.assertEqual(info.latest_version, manifest["version"])
        self.assertEqual(info.exe_url, manifest["downloadUrl"])
        self.assertEqual(info.sha256, manifest["sha256"])
        self.assertEqual(info.notes, manifest["notes"])
        self.assertEqual(info.latest_build_date, manifest["publishedAtUtc"])
        self.assertEqual(info.channel, "company")
        self.assertTrue(info.build_id_updates)

    def test_enterprise_latest_api_response_loads_manifest_asset(self) -> None:
        def fake_read(url: str, timeout: int) -> dict:
            return self.release if url == self.company.release_api_url else {"version": "0.1.1", "downloadUrl": "https://company/exe"}

        with patch.object(updater, "_read_json", side_effect=fake_read):
            info = updater._fetch_update_info_from_api(self.company)

        self.assertEqual(info.latest_version, "0.1.1")
        self.assertEqual(info.exe_url, "https://company/exe")

    def test_company_release_requires_exe_and_manifest_assets(self) -> None:
        release = {"assets": [{"name": "AngleCal.exe", "browser_download_url": "https://company/exe"}]}
        with patch.object(updater, "_read_json", return_value=release):
            with self.assertRaisesRegex(RuntimeError, "version.json"):
                updater._fetch_update_info_from_api(self.company)

    def test_version_comparison_and_notes_fallback(self) -> None:
        self.assertGreater(updater.compare_versions("1.10.0", "1.9.9"), 0)
        self.assertEqual(updater.compare_versions("1.0", "1.0.0"), 0)
        self.assertLess(updater.compare_versions("1.0.0", "1.0.1"), 0)
        info = updater._update_info_from_release(self.company, self.release, {"version": "0.1.1"}, None)
        self.assertEqual(info.notes, "Release notes")

    def test_sha256_mismatch_removes_download(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "update"
            info = updater.UpdateInfo("0", "", "", "1", "", "", "", "", "https://invalid", sha256="not-a-hash")
            with patch.object(updater, "update_work_dir", return_value=target), patch.object(updater.urllib.request, "urlopen") as urlopen:
                response = urlopen.return_value.__enter__.return_value
                response.headers.get.return_value = "3"
                response.read.side_effect = [b"abc", b""]
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    updater.download_update(info)
            self.assertFalse(list(target.glob("*.new.exe")))

    def test_same_update_notification_key_prevents_repeat_alerts(self) -> None:
        info = updater.UpdateInfo("0.1.0", "", "", "0.1.1", "", "", "", "", "https://company/exe", channel="company")

        self.assertTrue(updater.should_notify_update(info, None))
        self.assertFalse(updater.should_notify_update(info, info.notification_key))

    def test_same_company_version_detects_new_company_build_id(self) -> None:
        info = updater.UpdateInfo(
            "0.1.0", "company-old", "", "0.1.0", "company-new", "", "", "", "",
            channel="company", build_id_updates=True,
        )
        self.assertTrue(info.is_available)

    def test_company_channel_uses_enterprise_repository_name(self) -> None:
        self.assertIn("/bh2-min/AngleCal/", updater.COMPANY_CHANNEL.release_api_url)
        self.assertEqual(updater.COMPANY_CHANNEL.release_page_url, "https://github.samsungds.net/bh2-min/AngleCal/releases")

    def test_source_execution_disallows_exe_replacement(self) -> None:
        with patch.object(updater, "is_packaged_app", return_value=False):
            self.assertIn("소스 실행", updater.update_install_error())
            with self.assertRaisesRegex(RuntimeError, "소스 실행"):
                updater.launch_self_update(Path("new.exe"))

    def test_update_script_uses_backup_rollback_and_preserves_user_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.object(updater, "update_work_dir", return_value=Path(temp) / "work"):
            root = Path(temp)
            image = root / "sample.png"
            project = root / "sample.anglecal.json"
            image.write_bytes(b"image")
            project.write_text("project", encoding="utf-8")

            script = updater._write_update_script(root / "AngleCal.exe", root / "new.exe", 42)
            text = script.read_text(encoding="utf-8-sig")

            self.assertIn("$backupExe", text)
            self.assertIn("Move-WithRetry $targetExe $backupExe", text)
            self.assertIn("Move-Item -LiteralPath $backupExe -Destination $targetExe", text)
            self.assertNotIn(str(image), text)
            self.assertNotIn(str(project), text)
            self.assertTrue(image.exists())
            self.assertTrue(project.exists())

    def test_401_and_403_are_clear_authentication_errors(self) -> None:
        for status in (401, 403):
            error = urllib.error.HTTPError("https://company", status, "Denied", {}, None)
            with patch.object(updater.urllib.request, "urlopen", side_effect=error):
                with self.assertRaises(updater.UpdateAuthenticationError) as raised:
                    updater._read_url_bytes("https://company", 1)
            self.assertIn("권한", str(raised.exception))

    def test_write_permission_error_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            exe = Path(temp) / "AngleCal.exe"
            exe.write_bytes(b"exe")
            with patch.object(Path, "open", side_effect=PermissionError("denied")):
                message = updater.update_install_error(exe)
        self.assertIn("권한", message)
        self.assertIn("AngleCal.exe", message)

    def test_build_info_version_matches_pyproject(self) -> None:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(angle_cal.__version__, pyproject["project"]["version"])
        self.assertEqual(updater.build_info_dict()["app_version"], angle_cal.__version__)

    def test_company_channel_is_selected_from_embedded_build_info(self) -> None:
        with patch.object(updater, "UPDATE_CHANNEL", "company"):
            self.assertEqual(updater.selected_channel(), updater.COMPANY_CHANNEL)
        with patch.object(updater, "UPDATE_CHANNEL", "personal"):
            self.assertEqual(updater.selected_channel(), updater.PERSONAL_CHANNEL)

    def test_company_build_script_uses_spec_and_smoke_channel_check(self) -> None:
        script = Path("build-company-release.ps1").read_text(encoding="utf-8-sig")
        self.assertIn("AngleCal.spec", script)
        self.assertIn("--build-info-json", script)
        self.assertIn('update_channel -ne "company"', script)


if __name__ == "__main__":
    unittest.main()
