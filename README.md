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
2. **AngleCal Windows Latest** 릴리즈를 엽니다.
3. `AngleCal.exe`를 다운로드해서 더블클릭합니다.

`AngleCal.exe`는 Python 설치 없이 바로 실행하는 방식입니다.

### bat로 실행

1. GitHub repo의 **Releases**를 엽니다.
2. **AngleCal Windows Latest** 릴리즈를 엽니다.
3. `AngleCal-bat.zip`을 다운로드합니다.
4. 압축을 푼 뒤 `AngleCal.bat`를 더블클릭합니다.

`AngleCal.bat`는 첫 실행 때 `.venv`를 만들고 필요한 패키지를 설치한 뒤 앱을 실행합니다. Windows에 Python 3.9 이상이 설치되어 있어야 합니다.

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

**파일 → 애드온 → Trench 자동분석기**에서 엽니다. 기존 이미지·썸네일과 스케일 보정값을 가져오며,
분석 탭을 벗어나면 기존 측정 패널과 도구 상태를 복원합니다.

1. 기존 화면에서 입구면을 수평으로 맞추고 스케일바를 보정합니다. 스케일 없이도 px로 분석할 수 있습니다.
2. **영역 지정**을 누르고 **한 Trench**의 입구 높이부터 바닥 아래까지 드래그합니다. 양쪽 측벽 바깥의 재료도 조금 포함하세요. ROI 위쪽이 Depth의 기준 높이입니다.
3. CD 간격과 표시 단위(px / nm / Å)를 설정하고 **Trench 분석**을 누릅니다. nm/Å에는 nm/px 값이 필요하며 1 px 미만 간격은 보간값입니다.
4. 경계가 다르면 경계값·밝은 Trench·평활 범위를 조절해 다시 분석합니다. 휠로 확대, 드래그로 이동하며 오른쪽 전체 보기에서 원하는 위치를 누를 수 있습니다.
5. CD 표나 그래프를 누르면 그 깊이를, 국부 Bowing 표를 누르면 해당 구간을 원본에서 표시합니다. 그래프는 CD/깊이, 통합 윤곽, 좌우 Bowing으로 전환합니다.
6. **입구 곡률** 탭에서 좌우 대표 R과 사용 윤곽(노란 점), 적합 원호(보라)를 확인합니다. 검색 영역을 다시 그리거나 시작·끝점을 찍고 **구간 적용**을 누르면 해당 설정으로 다시 분석합니다. 초록/분홍 표시는 시작/끝입니다. **자동 초기화**는 해당 측벽의 보정만 초기화합니다.
7. **Export · GFE 좌표 복사**를 누르면 좌우 측벽과 Field가 GFE에 붙여넣을 수 있는 X/Y 두 열의 표로 클립보드에 들어갑니다. GFE 좌표 표에서 Ctrl+V로 붙여넣으세요. nm/px 보정이 필요하며 표시 단위에 관계없이 Å로 변환합니다. X=입구 중심 기준, Y=좌우 검출 Field의 입구 높이 평균 기준 음수 깊이입니다. 오른쪽 Field → 오른쪽 벽 → 바닥 → 왼쪽 벽 → 왼쪽 Field 순서로 전체 윤곽을 내보냅니다. 실제 검출된 좌우 Field 윤곽을 포함하며 윤곽 사이의 짧은 간격과 바닥 끝은 직선 연결합니다. Field가 미확정이거나 바닥이 잘린 영상은 복사하지 않습니다. 버튼 옆 화살표의 **파일 저장**으로 CSV(요약·곡률 진단/좌표·국부 구간·CD), JSON(설정·전체 경계 좌표·결과), PNG(주석 이미지와 좌우 입구 확대 그림)로 내보냅니다. 프로젝트·이미지 서식에는 좌우 검색 영역·원호 구간·곡률 결과가 포함됩니다. 다시 열면 저장 설정으로 분석을 재실행하며, 이미지 회전 시 영역과 곡률 설정은 재지정합니다.

| 측정값 | 계산 기준 |
| --- | --- |
| CD | 동일한 이미지 Y에서 오른쪽 경계 X − 왼쪽 경계 X |
| Depth | ROI 위쪽 입구 높이에서 명암 경계로 검출한 바닥까지의 수직 거리. 바닥 폐쇄가 확인되지 않으면 미확정으로 표시 |
| 전체 Bowing | 상단 검출점의 수직선 또는 상단–검출 깊이 90% 지점의 연결선을 기준으로 좌우 최대 외측 변위 |
| 국부 Bowing | 깊이가 증가할 때 외측으로 벌어지는 nega slope 구간을 찾고, 각 구간 시작 수직선에서 최대 외측 변위를 측정 |
| 국부 현 대비 휨 | 국부 시작점과 이후 복귀/극소점(없으면 분석 끝)을 잇는 직선 대비 최대 외측 변위. 직선 역경사와 볼록한 휨을 구분 |
| Nega 각도 | 평활화한 측벽의 외측 변위/깊이 기울기를 atan으로 변환한 수직 대비 각도. 최소 각도 설정으로 작은 잡음을 제외 |
| Field | ROI 입구 높이 주변의 재료/공간 명암 경계를 좌우에서 각각 추적하고 이상점을 제외해 표면 직선을 맞춤. 지지 길이·기울기·RMS와 원본 좌표를 기록 |
| 입구 각도 | 검출 Field 직선과 사용자가 선택한 깊이 구간의 측벽 직선 사이 내각. Field 기울기를 반영하며 곡면 한 점의 접선각과 다름 |
| 입구 곡률 반경 | 윗면에서 측벽으로 이어지는 둥근 전이 윤곽 전체에 이상점 영향을 줄이는 원 맞춤을 적용한 좌우 대표 R. 불안정하면 미확정과 이유를 표시 |

곡률은 전이 윤곽점 8개 이상, 원호 각도 45° 이상, RMS `max(1px, R의 5%)` 이하이며,
선택 구간 양끝을 각각 10% 줄였을 때 R 변화가 20% 이하여야 유효합니다.
평탄한 지지 구간의 명암 경계 퍼짐도 확인하며, R이 추정 퍼짐 폭의 두 배보다 작으면 흐림과 실제 곡률을 구분하기 어려워 미확정으로 표시합니다.
RMS는 윤곽과 원호 사이의 맞춤 오차이며 실제 치수 정확도나 신뢰 확률이 아닙니다.
입구 검색 크기와 좌우 곡률 보정은 입구 각도·CD·깊이·Bowing 계산에 영향을 주지 않습니다.

측벽과 Bowing은 바닥 곡면 영향을 줄이기 위해 검출된 프로파일의 마지막 10%를 제외합니다.
입구의 경계가 끊기면 첫 유효 경계부터 CD와 Bowing을 계산하고 경고를 표시합니다.
측벽은 명암으로 Trench 후보를 추적한 뒤, Trench 안쪽에서 재료 방향으로 연결된 첫 밝은 테두리 정점을 원본 밝기에서 찾습니다. 전체 재료의 가장 밝은 점이나 떨어진 층 경계는 선택하지 않습니다. 정점은 최대 18 px 안에서 찾고, 분명한 밝은 테두리가 없으면 연결된 명암 변화의 중심을 사용해 Field와 측벽의 기준을 통일합니다. 평활 범위 설정은 최종 윤곽에도 적용됩니다. JSON에는 측벽 검출 방식도 기록합니다.

경계 위치는 영상 대비·노이즈·해상도에 영향을 받습니다. 밝기 정점은 영상상의 측정 기준이며 실제 재료 경계의 정확도를 보증하지 않습니다. 자동 기울기 보정이나 SEM 촬영 기하 보정은 수행하지 않으며,
실제 측정 전 표시된 경계·입구 높이·스케일을 확인해야 합니다. 생성된 SEM 샘플은 계측 정확도 기준으로 쓰지 않습니다.

검증 화면과 합성 샘플 내보내기는 `scripts/verify_trench_workspace.py`로 재현할 수 있습니다.

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

Trench 그래프는 기본으로 **통합 윤곽**을 표시합니다. 왼쪽 벽(청록)과 오른쪽 벽(주황)을 같은 X축에 그리므로 벽 사이 간격이 해당 깊이의 CD입니다. 분홍 치수선에 CD를 표시하며 그래프·표에서 깊이를 선택하면 함께 이동합니다. CD/깊이와 좌우 Bowing 그래프도 선택할 수 있습니다. 가로·세로 축은 화면에 맞춰 각각 표시하므로 이미지의 1:1 종횡비를 뜻하지 않습니다.

**Field / 각도** 탭에는 좌우 확대 화면과 측정 근거가 표시됩니다. 초록은 검출 Field, 노란 점/점선은 각도에 사용한 측벽과 연장선, 분홍은 교점과 각도 호입니다. **Field 검색 ±px**는 ROI 상단 주변 검색 범위이고 **각도 측벽 시작/끝 px**는 각 측벽 입구 Field 높이에서 내려간 깊이입니다. 기본 40~100px를 사용하며 직선 측벽 구간으로 조정하세요. Field 지지점 16개 이상·60% 이상 공간 범위와 RMS 1.5px 이하를 요구합니다. 각도는 지지점 16개 이상·깊이 범위 15px 이상·RMS 1.5px 이하 및 구간 양쪽 기울기 차이 5° 이하일 때 표시합니다. 표면이나 직선 측벽이 불충분하면 이유와 미확정을 표시합니다. RMS는 맞춤 오차이고 실제 치수 정확도가 아닙니다. Field와 각도 설정은 곡률 검색과 독립이며 CD·Depth·Bowing을 바꾸지 않습니다. 프로젝트·CSV·JSON에 측정 구간/좌표/직선/오차/판정을 저장하고 PNG에 표면·각도 표시를 포함합니다. 기존 파일은 기본 범위로 열며 재분석합니다.

Field / 각도 탭은 측정 Field와 cliff 직선에 동시에 접하는 원호를 반복 적합해 접선 R, 원호 RMS, 접점, 중심과 반지름 선을 표시합니다. Field→원호→cliff가 위치와 접선 방향 모두 이어집니다. 그래프와 GFE Export도 유효한 접선 원호를 사용합니다. 독립 입구 곡률 탭의 대표 R과 접선 R이 허용 범위 밖으로 다르거나 점수·해상도·안정성 기준을 충족하지 않으면 연결 원호는 미확정으로 남깁니다. 이때 각도는 직선 내각으로 표시하며 장식용 호를 실제 곡률처럼 그리지 않습니다. JSON/CSV 및 프로젝트에 connection 모델과 원호 좌표를 저장합니다.
