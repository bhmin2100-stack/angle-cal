# Angle Cal

Vertical SEM 이미지에서 선분 기반 각도를 측정하는 Windows용 데스크톱 도구입니다.

## 주요 기능

- SEM 이미지 불러오기: PNG, JPG, BMP, TIFF 지원
- 폴더 브라우저: 폴더와 하위 폴더의 이미지를 트리 기준으로 불러오고 오른쪽 썸네일 뷰에서 폴더별 구획과 1열/2열 썸네일 표시
- 이미지 전환: 썸네일 또는 `PageUp`/`PageDown`으로 이미지 전환 시 스케일바 캘리브레이션 비율 유지
- 스케일바 캘리브레이션: 스케일바 위에 선을 긋고 실제 nm 값을 입력하면 nm/px 계산
- 스케일 프리셋: 캘리브레이션 비율을 1~9번 슬롯에 등록하고 숫자키로 적용, 순서 변경/편집/삭제 지원
- 기준선 정렬: 이미지마다 기준선 하나를 두고, 수평기준선/수직기준선 토글 후 이미지 전체를 기준축에 맞춰 회전
- 경계선 형태: 단일 직선 경계선과 PowerPoint식 점 입력 세그먼트 경계선을 선택해 작성
- 경계선 인식: 사용자가 대략 그은 경계선을 수직 방향으로 탐색해 명도 변화량이 가장 큰 위치로 이동하고, 선택한 형태대로 단일 직선 또는 직선 세그먼트 체인으로 보정
- 경계인식 범위 표시: 경계선 주변의 인식 반경을 px 단위로 설정하고 이미지 위에 반투명 초록색 영역과 범위 숫자를 각각 표시/숨김
- 세그먼트 크기: 인식된 경계를 구성하는 세그먼트 단위를 조절하고 이미지 우상단 샘플로 확인
- 각도 계산: 기준선 대비 경계선 각도와, 일정 간격 가이드 선분과 경계선의 교점 각도 계산
- 각도 표시 편집: 선택한 경계선별로 교점의 1~4사분면 각도 호 위치, 호 크기, 숫자 위치/거리 조정
- CD 길이 측정: 한 줄의 가이드와 여러 경계선 교점 사이 길이를 전체/홀수번째/짝수번째 구간으로 표시
- 구조 등록: 자주 쓰는 경계선/가이드/각도/CD 분석 양식을 이름으로 저장하고 `Ctrl+Shift+C`/`Ctrl+Shift+V`로 재사용, 구조 파일 공유 지원
- 표시 편집: `Esc`로 선택 도구 전환, 드래그 박스 선택, `Q`/`W`/`E` 선택 필터, 방향키 이동, `Ctrl+방향키` 1px 이동, `Delete` 삭제, `Ctrl+C`/`Ctrl+V` 상위개체 복사/붙여넣기 지원
- 크기 조절: 개체를 선택한 뒤 **크기 조절** 도구에서 드래그해 선/세그먼트/가이드/각도 묶음 확대·축소
- 상단 도구바: 파일, 내보내기, 도구, 기준, 인식, 가이드 묶음을 얇은 핸들로 이동 배치 가능
- 경계선 계층: 경계선이 상위개체이고, 그 경계 때문에 생긴 각도 숫자/호는 하위개체로 관리
- 표시 레전드: 오른쪽 아래 **표시** 체크박스에서 스케일바, 기준선, 경계, 가이드, 각도, CD 길이, 경계 길이, 인식 범위 영역, 인식 범위 숫자를 묶음 단위로 숨김/표시
- 탐색 편의: `Ctrl + 마우스 휠` 확대/축소, 이동 도구 또는 마우스 오른쪽/가운데 드래그로 화면 이동, `PageUp`/`PageDown` 이미지 전환
- 내보내기: 주석이 포함된 PNG, 측정값 CSV, 프로젝트 JSON 저장/열기

## Windows에서 실행

두 가지 방식 중 편한 쪽으로 실행할 수 있습니다.

### exe로 실행

1. GitHub repo의 **Releases**를 엽니다.
2. **최신 Windows 릴리스**를 엽니다.
3. `AngleCal.exe`를 다운로드해서 더블클릭합니다.

`AngleCal.exe`는 Python 설치 없이 바로 실행하는 방식입니다.

### bat로 실행

1. GitHub repo의 **Releases**를 엽니다.
2. **최신 Windows 릴리스**를 엽니다.
3. `AngleCal-bat.zip`을 다운로드합니다.
4. 압축을 푼 뒤 `AngleCal.bat`를 더블클릭합니다.

`AngleCal.bat`는 첫 실행 때 `.venv`를 만들고 필요한 패키지를 설치한 뒤 앱을 실행합니다. Windows에 Python 3.9 이상이 설치되어 있어야 합니다.

**[최신 EXE 직접 다운로드](https://github.com/bhmin2100-stack/angle-cal/releases/download/windows-latest/AngleCal.exe)** · **[최신 배치 실행 패키지](https://github.com/bhmin2100-stack/angle-cal/releases/download/windows-latest/AngleCal-bat.zip)**

릴리스 이름과 창 제목의 버전·빌드 번호를 비교하세요. v0.1.6의 Trench 화면에는 **곡선 그리기 · 구조 불러오기 · 인식** 전용 도구가 표시됩니다. `Source code`는 실행 파일이 아니라 소스이며, 릴리스 태그도 해당 EXE를 만든 커밋으로 함께 갱신합니다.

직접 exe를 만들려면 Python 3.10 이상이 설치된 Windows에서:

```bat
build_windows.bat
```

빌드가 끝나면 `dist\AngleCal.exe` 하나가 생성됩니다. 이 파일만 다른 폴더나 PC로 옮겨 실행하면 됩니다.

개발 모드로 바로 실행하려면:

```bat
AngleCal.bat
```

또는:

```bat
py -3 -m pip install -e .
python run_angle_cal.py
```

## 기본 사용 흐름

1. **이미지 열기**로 SEM 이미지를 하나 불러오거나, **폴더 열기**로 폴더와 하위 폴더의 이미지를 함께 불러옵니다.
2. 폴더를 열면 오른쪽 **썸네일** 뷰에서 폴더별 구획과 이미지 목록이 표시됩니다. 썸네일 뷰 상단에서 1열/2열 배치를 바꿀 수 있고, `PageUp`/`PageDown`으로 이전/다음 이미지로 전환할 수 있습니다. 이미지 전환 시 기존 스케일바 비율은 유지됩니다.
3. 상단 도구바에서 **스케일바** 버튼을 누르고 스케일바 양 끝을 드래그한 뒤 실제 길이(nm)를 입력합니다.
4. 자주 쓰는 비율은 오른쪽 **스케일 프리셋**에서 **현재 등록**으로 저장합니다. 등록 순서가 숫자키 `1~9` 적용 순서입니다.
5. **기준선** 버튼을 누르고 기준 성분을 따라 선을 긋습니다. 기준선은 이미지 안에서 하나만 유지되고 50% 투명도로 표시됩니다.
6. 기준 종류를 **수평기준선** 또는 **수직기준선**으로 고른 뒤 **이미지 맞춤**을 누릅니다. 기준선 도구가 선택된 상태에서 토글을 바꾸고 다시 **이미지 맞춤**을 누르면 같은 기준선이 새 축 기준으로 정렬됩니다.
7. **경계 형태**에서 **직선** 또는 **세그먼트**를 고릅니다. 직선은 드래그로 그리고, 세그먼트는 점을 찍은 뒤 더블클릭 또는 `Enter`로 확정합니다.
8. **경계선** 버튼을 누르고 측정할 구조 경계를 대략 따라 그립니다.
9. **인식 설정**에서 경계인식 범위와 세그먼트 크기를 조절합니다. 값을 바꾸면 이미지 우상단에 샘플이 표시되고, **범위 표시**가 켜져 있으면 이미지 위에 반투명 초록색 경계인식 영역과 px 폭이 표시됩니다. 오른쪽 아래 **표시**에서 영역과 숫자를 각각 끌 수 있습니다.
10. **인식**을 누르면 경계선이 설정된 경계인식 범위 안에서 명도 변화 최대 위치를 따라 선택한 형태로 보정됩니다.
11. 필요하면 가이드 방향/간격을 정하고 **그리기**를 누릅니다.
12. **각도 계산**으로 기준선 대비 각도와 가이드 교점 각도를 표시합니다.
13. CD 길이는 `CD 전체`, `CD 홀수번째`, `CD 짝수번째` 중 하나를 고른 뒤 **CD 측정**을 누르면 가이드선과 경계선 교점 사이 길이가 표시됩니다.
14. 자주 쓰는 분석 양식은 경계선을 선택한 뒤 **구조 저장** 또는 `Ctrl+Shift+C`로 이름을 붙여 저장합니다. 선택 경계선이 없으면 현재 이미지의 모든 경계선과 가이드를 구조로 저장합니다.
15. 저장된 구조는 **구조** 드롭다운에서 고른 뒤 **구조 붙여넣기** 또는 `Ctrl+Shift+V`로 불러옵니다. 현재 이미지에 가이드가 이미 있으면 구조 안의 가이드는 중복으로 불러오지 않습니다.
16. **구조 공유**는 선택한 구조를 `.anglecal.structure.json` 파일로 저장하고, **구조 가져오기**로 다시 등록할 수 있습니다.
17. 각도 위치나 크기를 바꾸려면 경계선을 하나 또는 여러 개 선택하고 **각도 표시 편집**을 누릅니다. 교점 주변 1~4사분면 중 하나를 고르고, 호 크기와 숫자 위치/거리를 조정할 수 있습니다.
18. `Esc`를 누르면 선택 도구로 돌아갑니다. 드래그 박스로 여러 개체를 선택할 때 `Q`를 누른 채 드래그하면 경계선만, `W`는 각도 호만, `E`는 각도 숫자만 선택됩니다. 선택 개체는 드래그 또는 방향키로 옮길 수 있고, `Ctrl+방향키`는 1px씩 이동합니다. `Delete`로 삭제할 수 있고, `Ctrl+C`/`Ctrl+V`로 경계선 같은 상위개체를 복사/붙여넣기할 수 있습니다. 각도 숫자와 호 같은 하위개체는 복사 대상에서 제외됩니다.
19. 선택한 개체는 **크기 조절** 도구에서 드래그해 확대·축소할 수 있습니다. 각도 숫자와 호는 하나의 묶음으로 같이 선택됩니다.
20. 오른쪽 아래 **표시** 레전드에서 묶음 단위로 보이기/숨기기를 전환할 수 있습니다.
21. 결과는 **CSV 내보내기** 또는 **주석 PNG 내보내기**로 저장합니다.

## 경계 인식 방식

사용자가 그은 경계선을 중심으로 선분에 수직인 방향의 픽셀 밝기 프로파일을 만듭니다. 경계인식 범위는 **인식 설정** 또는 툴바의 **경계인식 범위 px** 값으로 조절합니다. 예를 들어 20 px이면 경계선 양쪽 20 px, 총 40 px 폭 안에서만 명도 변화를 찾습니다. 각 오프셋 위치에서 선분 주변의 평균 명도를 샘플링하고, 이 1차원 프로파일의 기울기 절댓값이 최대인 오프셋을 실제 경계로 판단합니다. **직선** 모드는 경계선 전체를 하나의 선분으로 유지하면서 이동하고, **세그먼트** 모드는 여러 직선 세그먼트별로 경계를 찾아 체인처럼 이어진 경계선으로 따라갑니다. 세그먼트 경계는 기준선 대비 각도도 세그먼트별로 계산됩니다. **세그먼트 크기**가 높을수록 더 촘촘한 점과 약한 smoothing으로 국소 경계를 민감하게 따라가고, 낮을수록 더 긴 직선 세그먼트 체인이 됩니다. 선택한 선분만 인식하려면 선분을 먼저 선택한 뒤 **인식**을 누르면 됩니다. 아무 선분도 선택하지 않으면 모든 경계선이 인식됩니다.

## 개발자용

### Trench 자동분석기

**파일 → 애드온 → Trench 자동분석기**에서 엽니다. v0.1.6부터 다음 순서로 사용합니다.

1. **썸네일에서 이미지 선택**
2. **곡선 그리기**로 인식할 벽을 따라 점을 찍고 마지막에 더블클릭합니다. **구조 불러오기**로 저장한 곡선이나 기존 AngleCal 경계 구조를 불러올 수도 있습니다.
3. **인식**을 누릅니다. 분석 영역과 입구 높이, 명암 방향은 자동으로 찾습니다.

Trench 화면에서는 전용 도구줄의 **선택 / 이동 / 크기 조절 / 스케일바 / 곡선 그리기 / 구조 불러오기 / 구조 저장 / 인식**을 사용합니다. v0.1.8부터 **선택 도구에서 기존 점 사이의 곡선을 클릭하면 새 점이 추가**됩니다. 클릭한 채 바로 끌거나 추가된 점을 다시 끌어 곡선을 보정할 수 있습니다. 기존 점을 클릭하면 점을 추가하지 않고 그 점을 편집합니다. **Alt+드래그**는 곡선 전체 이동이며 이동 도구는 화면을 이동합니다. 곡선을 그릴 때는 화살표 커서를 사용합니다. 크기 조절 도구에서 모서리 핸들을 드래그하면 곡선 크기를 조절합니다. 휠은 인식 폭, Ctrl+휠은 화면 확대·축소입니다. Backspace는 그리는 중 마지막 점 취소, Esc는 취소입니다.

새 기본 인식은 **표시된 곡선 인식 범위 안에서만** 경계를 선택합니다. 양쪽 벽과 상단 Field, 바닥까지 원하는 부분을 곡선 범위에 포함하세요. 한쪽 벽만 그렸을 때 반대쪽 벽을 범위 밖에서 자동으로 채우거나 곡선 끝을 넘어 경계를 연장하지 않습니다. Field·바닥을 포함하지 않았다면 해당 측정은 미확정입니다. CD 보간 좌표와 GFE 연결 형상도 같은 범위 제한을 검사합니다.

v0.1.11부터 인식 결과가 표시되면 가이드 곡선·인식 띠·이름을 불투명도 18%로 옅게 표시합니다. 검출된 경계는 그대로 유지합니다. 가이드를 선택해 편집할 때는 다시 선명하게 보이며, 수정으로 결과가 초기화되면 모든 가이드가 원래 표시로 돌아옵니다.

v0.1.12의 기본은 **밝은 테두리 기준**입니다. 주변 잡음을 줄인 밝기 분포의 최고점을 추적한 뒤, 측정 윤곽에 수직으로 **빈 공간 쪽 1px** 이동합니다. 왼쪽 벽은 오른쪽으로, 오른쪽 벽은 왼쪽으로 이동하며 Field·바닥에서도 빈 공간 방향을 판별합니다. `빈 공간 쪽 이동 (px)`에서 0~10px로 조절하며 0은 최고점 자체입니다. 명암이 단조롭게 번지는 경사면을 밝은 띠 대신 선택하지 않습니다. 밝은 띠가 없는 영상은 체크를 끄면 기존 가이드 인식 기준을 사용할 수 있습니다. 이동량은 원본 픽셀 기준으로 저장되고 단위 변경에 영향받지 않습니다. 이는 일관된 경계 정의이며 실제 치수 정확도를 보증하는 보정값이 아닙니다.

일반 이미지 화면과 Trench 화면 모두 **Ctrl+왼쪽 드래그**로 사진을 이동할 수 있습니다. 곡선 그리기·편집 도중에도 점을 바꾸지 않고 화면만 이동하며, Ctrl+휠 확대는 유지합니다.

짧게 끊긴 경계는 앞뒤에서 검출한 윤곽의 방향을 따라 부드럽게 연결합니다. 최대 12px의 가이드 길이 안에서 양끝 지지가 있고, 연결선 전체가 인식 범위 안에 있을 때만 보간합니다. 이미지·그래프·CD·GFE가 같은 연결 윤곽을 사용하며, 보간 구간의 개수는 화면에, 좌표와 구분은 프로젝트·JSON·CSV에 기록합니다. 긴 공백이나 서로 다른 경계로 의심되는 구간은 연결하지 않습니다.

v0.1.9부터 SEM의 넓은 음영 변화는 곡선 주변의 명암과 실제 연결 윤곽으로 판별하며, 떨어진 정보 띠가 명암 기준을 바꾸지 않도록 자동 검색 문맥을 제한합니다. 치수 글자·화살표나 약한 경계 때문에 일부 구간이 미검출이면 화면에 표시하고 해당 CD와 GFE 연결을 제한합니다. 검출률은 실제 치수 정확도를 뜻하지 않으므로 기준 치수와 선택된 윤곽 위치를 함께 확인하세요.

v0.1.10부터 기본 Trench 인식은 곡선을 따라 이웃한 영상의 경계 근거를 모아 입자 잡음과 단발성 주석선 영향을 줄입니다. 한 행의 가장 강한 점만 고르지 않고 연결된 후보 경로를 평가하며, 실제 보우·기울기·홈·둥근 입구를 유지하도록 검사합니다. 직선으로 강제 보정하거나 가이드 좌표를 측정값으로 쓰지 않습니다. 영상 근거가 없거나 후보가 충돌하는 구간은 미검출로 남기며 GFE 제한도 유지합니다. 이 처리는 형상 인식을 위한 것으로 미세 거칠기 계측이나 치수 정확도 보증이 아닙니다. 방법과 한계는 [윤곽 연속성 설계](docs/trench-continuity.md)를 참고하세요.

기존 이미지에 스케일 보정값이 있으면 그대로 이어받습니다. Trench 안에서도 **스케일바**로 선을 그린 뒤 실제 nm 길이를 입력할 수 있습니다. nm/px와 스케일 선은 기존 AngleCal 이미지·프로젝트에 함께 저장됩니다. 보정하지 않은 영상은 px로 표시합니다.

**CD 깊이 간격은 기본 1nm**이며, 표시 단위를 px/nm/Å로 바꿔도 측정 간격은 변하지 않습니다. 원본 1px보다 작은 간격은 윤곽 보간값입니다. 최대 100,000개 지점을 내보낼 수 있습니다. 양쪽 벽은 같은 X축 그래프에 표시되므로 벽 사이 간격이 CD입니다. v0.1.7부터 상단 Field와 입구 전이, 최하단 바닥까지 전체 윤곽을 함께 표시합니다. v0.1.11의 짧은 누락 연결 조건을 만족하지 못한 구간은 공백으로 남으며, 바닥 윤곽 없이 깊이만 검출된 경우 바닥 기준을 점선으로 표시합니다. 표나 그래프를 누르면 해당 깊이를 이미지에서 확인합니다.

화면에는 보정값, CD 간격, 인식, 전체 보기, 깊이/CD/Bowing 결과와 Export만 표시합니다. 영역 지정·경계값·명암·Bowing 기준·Nega 각도·입구/Field 검색·각도 측벽 구간·그래프 모드 설정, 입구 곡률/Field·각도 탭은 표시하지 않습니다. 기본 가이드의 인식 폭과 경계 기준(기울기 최대/밝은 꼭대기 등)을 이어받으며, 영상의 실제 경계를 곡선 범위 안에서 찾습니다.

**구조 저장**은 곡선과 인식 폭을 `.anglecal.structure.json`에 저장합니다. 불러온 곡선도 선택·이동·크기 조절 후 바로 인식할 수 있습니다. 구조 불러오기 옆 화살표에서는 기존 구조 목록을 선택할 수 있습니다.

**Export · GFE 좌표 복사**를 누르면 Å 단위 X/Y 표가 클립보드에 복사됩니다. GFE에서 Ctrl+V로 붙여넣으세요. 보정값과 완전한 경계가 필요하며, 끊겼거나 바닥이 잘린 윤곽은 복사를 제한합니다. 옆 화살표의 파일 저장은 CSV/JSON/PNG를 지원합니다. PNG에는 현재 분석 경계와 간단한 결과를 저장합니다. 기존 프로젝트/구조는 계속 열 수 있으며 기존 세부 계측 데이터는 JSON/CSV 호환성을 유지합니다.

합성 영상의 화면 조작·스케일·썸네일 전환·구조 불러오기·내보내기는 `scripts/verify_trench_simple_workflow.py`로 확인합니다. 실제 SEM 치수 정확도는 기준값이 있는 영상으로 별도 검증해야 합니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
QT_QPA_PLATFORM=offscreen .venv/bin/python -c "from PySide6.QtWidgets import QApplication; from angle_cal.app import MainWindow; app=QApplication([]); print(MainWindow().windowTitle())"
```


## Photo merge registration

Automatic alignment searches translations independently of board placement. Each refined
candidate is scored using the entire valid overlap: 70% normalized brightness correlation
and 30% signed 2-D edge correlation. Normalization tolerates brightness gain/offset changes;
no individual band or sparse pixel sample determines the final translation score.
Cropped areas and detected instrument footers are excluded. Candidates need at least
22% of the smaller valid image area in common. Feature-based affine/perspective fallback
also validates the full overlap, with a penalty for more complex transforms.

Band graphs are diagnostic only: source previews mark the sampled regions and real image
strips share the graph's horizontal axis. Nearly tied whole-area matches and almost complete
overlays use the highest unrounded candidate score and display a nonblocking warning after
merging. Exact ties prefer more overlapping pixels. Selected-pair warnings are saved in the
JSON report. Disconnected sets without a usable match still cannot be composited.
Native tonal differences (including 16-bit input) are scored in float64 without 8-bit
quantization; up to 32 coarse peaks receive full-resolution integer-position refinement.
Scores have no 99% cap and are displayed to two decimal places (including 99.99% or 100.00%).
Reported percentages are heuristic quality scores, not probabilities or a guarantee of
correct physical placement; subpixel translation is not searched by this path.

Candidate positioning uses a 480 px global search followed by sampled 1600 px and native
integer refinement. Each resulting candidate is then scored exactly once using every valid
native overlap pixel. This bounds expensive full-resolution verification at 32 passes per
image pair while retaining native-depth final scores.

The merge board's outer-edge option is enabled by default. Registration still uses the
selected crop rectangles, while composition restores the area above the topmost image's
crop and below the bottommost image's crop. Horizontal crop bounds and every interior
image crop remain in effect. Turning the option off applies the crop masks to every edge.

Board controls: click empty board space to finish cropping while retaining the selection;
arrow keys move selected images by 10 board pixels, Ctrl+arrow keys by 1. Delete removes
only selected board items. Add-ons open as menu tabs in the main window.

이미지 서식에서 지정한 명도·대비·선명도 보정은 Trench 이미지와 전체 보기, 경계 인식에 함께 반영됩니다. 보정값을 바꾸면 이전 분석 결과는 지워지며, 기존 곡선과 스케일을 유지한 상태에서 다시 인식할 수 있습니다. 이미지별 보정값은 프로젝트와 이미지 서식 저장 시 함께 보존됩니다.

### Trench 윤곽 보정 (v0.1.14)

오른쪽 **윤곽 보정**에서 이동평균·가우시안·Savitzky–Golay·강건 스플라인 중 하나를 선택합니다. 각 방법의 강도(0~100)는 따로 기억되며 한 방법씩 적용됩니다. 0은 추가 보정 없이 기존 분석과 같습니다. 인식 후 슬라이더를 놓거나 키보드 조절을 멈추면 자동으로 재분석합니다. **보정 전 경계 겹쳐 보기**로 회색 점선과 비교하고, **보정 초기화**로 모든 방법의 강도를 0으로 되돌립니다.

전체 검출 윤곽을 곡선 길이 기준으로 처리하므로 Field·양쪽 벽·바닥에 같은 방식이 적용됩니다. 강도는 원본 좌표의 약 3~41px 범위와 최대 0~4px 이동 제한으로 연결됩니다. 강건 스플라인은 반복 가중치를 쓰는 이산 2차 차분 평활화입니다. 끝점과 급한 방향 전환은 보정을 약하게 하며, 가이드 밖으로 나가는 이동은 적용하지 않습니다. 남아 있는 미검출 구간을 새로 연결하지 않고, 매번 보정 전 검출 결과에서 다시 계산합니다. 모든 미세 요철을 잡음으로 구별하거나 모서리·치수를 완전히 보존한다는 뜻은 아닙니다.

보정 윤곽에서 CD·Bowing·Field/각도를 다시 계산하며 그래프와 내보내기도 동일한 결과를 사용합니다. **Depth의 바닥 폐쇄 판단은 기존 밝기 기준을 유지합니다.** 프로젝트에는 방법별 강도와 비교 표시를, JSON/CSV에는 적용 방법·강도·보정 전후 윤곽·실제 이동량을 저장합니다. 기존 프로젝트는 추가 보정 0으로 열립니다. 단위 변경은 픽셀 기준 강도를 바꾸지 않습니다.
