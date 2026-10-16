"""Publish the same version/build identity embedded by AngleCal.spec."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from angle_cal.release_metadata import embedded_metadata, release_manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exe',default='dist/AngleCal.exe')
    parser.add_argument('--build-info',default='build/_build_info.json')
    parser.add_argument('--output',default='dist/version.json')
    args=parser.parse_args()
    metadata=embedded_metadata(args.build_info)
    if not metadata:parser.error('Missing or invalid packaged build metadata')
    manifest=release_manifest(args.exe,metadata,
                              'https://github.com/bhmin2100-stack/angle-cal/releases/download/windows-latest/AngleCal.exe')
    Path(args.output).write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Version {manifest['version']} / build {manifest['build_id']}")


if __name__=='__main__':main()
