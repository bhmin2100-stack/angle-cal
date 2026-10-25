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
