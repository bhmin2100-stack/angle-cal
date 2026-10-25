# Trench 인식 기준 샘플

사용자가 2026-10-25 지정한 **항상 사용하는 인식 검증 샘플**은
`trench-reference.png`이다. 이후 Trench 인식·Field·깊이·TCD·보정·내보내기
변경 시 이 이미지의 왼쪽과 오른쪽 트랜치를 각각 확인한다.

- 원본: 사용자가 첨부한 `SYNTHETIC SAMPLE`, `SAMPLE-04`, `INNER-REAL-01` 이미지
- 크기: 1536 × 1024 px
- SHA-256: `b54152df5fe770c925c9d5d5a1f11b952dfe31fe123d74085365321ed6380c99`
- 원본을 수정하거나 재생성하지 않았다. 합성 영상이므로 실제 SEM의 치수 정확도 검증용 기준값은 아니다.

이미지를 연 뒤 Trench 분석기에서 `trench-reference-left.anglecal.structure.json`
또는 `trench-reference-right.anglecal.structure.json`을 구조 불러오기로 선택한다.
두 구조는 각각 양쪽 Field와 벽·바닥을 포함한 하나의 가이드다.

스크린샷의 0~10 µm 눈금은 대략 (1124,973)~(1437,973)이다.
이 눈금으로 UI 동작을 확인할 때는 313 px = 10,000 nm, 약 31.9489 nm/px를
사용했다. 이는 인쇄된 합성 눈금의 수동 읽기이며 실제 장비 교정값이 아니다.
구조 파일은 곡선만 불러오므로 스케일바 또는 기존 보정값을 별도로 설정한다.

수치 오차 검사는 별도로 알려진 형상을 만든 합성 단위 테스트에서 수행한다.
이 기준 샘플의 검증 산출물은 `build/region-continuity-verification/reference/`에
이미지별 CSV·JSON·PNG·TSV 및 `trench-profile/v1` 교환 JSON으로 저장한다.

## TEM 추가 검증 샘플

`trench-tem-reference.png`는 사용자가 2026-10-25 추가한 SAMPLE-05 / TiO2-SPACER-01
합성 TEM 이미지의 원본 사본이다. VSEM 기준 샘플을 대체하지 않고 함께 검증한다.
SHA-256은 `b29fe1e3bcec489617033785665ed2d4530a34fd24496e13d28bab8dda6fc722`이다.

사용자가 지정한 경계 기준은 **검은 띠를 제외하고 회색 재료가 끝나는 쪽**이다.
검은 띠의 중심이나 흰 빈 공간 쪽 끝점을 기준으로 하지 않는다.
`trench-tem-reference-left.anglecal.structure.json`과 `...-right.anglecal.structure.json`을
각각 불러오고 영상 방식을 자동 또는 TEM으로 선택한다. 두 구조 모두 Field와 바닥을 포함한다.
`scripts/verify_trench_tem_reference.py`는 실제 Qt 분석 패널에서 양쪽을 인식하고
`build/tem-verification/reference/`에 화면·CSV·JSON·PNG·GFE 좌표를 저장한다.

표시 확인용 스케일은 원본에 인쇄된 10 µm / 약 313 px를 사용한다. 이는 합성 이미지의
눈금이며 실측 정확도를 보증하지 않는다. 알려진 띠 두께 3·8·14 px, 잡음·흐림·16비트·
역방향 가이드·명암 반전은 별도 수치 테스트로 확인한다.
