# Angle Cal

Vertical SEM 이미지에서 선분 기반 각도를 측정하는 Windows용 데스크톱 도구입니다.

## 주요 기능

- SEM 이미지 불러오기: PNG, JPG, BMP, TIFF 지원
- 스케일바 캘리브레이션: 스케일바 위에 선을 긋고 실제 nm 값을 입력하면 nm/px 계산
- 기준선 정렬: 수평 또는 수직 기준선을 그린 뒤 이미지 전체를 기준축에 맞춰 회전
- 경계선 인식: 사용자가 대략 그은 경계선을 선분의 수직 방향으로 탐색해 명도 변화량이 가장 큰 위치로 이동
- 각도 계산: 기준선 대비 경계선 각도와, 일정 간격 가이드 선분과 경계선의 교점 각도 계산
- 표시 편집: 각도 숫자 라벨은 선택해서 원하는 위치로 이동 가능
- 탐색 편의: `Ctrl + 마우스 휠` 확대/축소, 이동 도구 또는 마우스 오른쪽/가운데 드래그로 화면 이동
- 내보내기: 주석이 포함된 PNG, 측정값 CSV, 프로젝트 JSON 저장/열기

## Windows에서 실행

Python 3.10 이상이 설치되어 있다면:

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m angle_cal
```

또는 소스 폴더에서 바로 실행하려면:

```bat
py -3 -m pip install -e .
python run_angle_cal.py
```

실행 파일을 만들려면:

```bat
build_windows.bat
```

빌드가 끝나면 `dist\AngleCal\AngleCal.exe`가 생성됩니다.

GitHub Actions의 **Build Windows** 워크플로를 수동 실행해도 Windows 실행 파일 아티팩트를 받을 수 있습니다.

## 기본 사용 흐름

1. **이미지 열기**로 SEM 이미지를 불러옵니다.
2. 도구를 **스케일바**로 바꾸고 스케일바 양 끝을 드래그한 뒤 실제 길이(nm)를 입력합니다.
3. 도구를 **기준선**으로 바꾸고 수평/수직 기준 성분을 따라 선을 긋습니다.
4. 기준 종류를 **수평 기준** 또는 **수직 기준**으로 고른 뒤 **이미지 맞춤**을 누릅니다.
5. 도구를 **경계선**으로 바꾸고 측정할 구조 경계를 대략 따라 그립니다.
6. **인식**을 누르면 경계선이 명도 변화 최대 위치로 스냅됩니다.
7. 필요하면 가이드 방향/간격을 정하고 **그리기**를 누릅니다.
8. **각도 계산**으로 기준선 대비 각도와 가이드 교점 각도를 표시합니다.
9. 결과는 **CSV 내보내기** 또는 **주석 PNG 내보내기**로 저장합니다.

## 경계 인식 방식

사용자가 그은 경계선을 중심으로 선분에 수직인 방향의 픽셀 밝기 프로파일을 만듭니다. 각 오프셋 위치에서 선분을 따라 평균 명도를 샘플링하고, 이 1차원 프로파일의 기울기 절댓값이 최대인 오프셋을 실제 경계로 판단합니다. 선택한 선분만 인식하려면 선분을 먼저 선택한 뒤 **인식**을 누르면 됩니다. 아무 선분도 선택하지 않으면 모든 경계선이 인식됩니다.

## 개발자용

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
QT_QPA_PLATFORM=offscreen .venv/bin/python -c "from PySide6.QtWidgets import QApplication; from angle_cal.app import MainWindow; app=QApplication([]); print(MainWindow().windowTitle())"
```
