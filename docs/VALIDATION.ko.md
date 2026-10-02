# 검증 범위와 artifact 관계

기준 기기: M5 Max GPU40코어/128GiB, macOS27.0/26A428, AC 전원. 실제 코딩 측정은24/24셀을 판정했으며 각 셀에 고정된 작업 유형3종을 사용했습니다. 대표2Ki C1/C4는 n5입니다. [전체 표](BENCHMARKS.ko.md), [실제 Pi 기록](DEMO.ko.md), [측정 방법](MEASUREMENT_METHOD.md), [CSV](data/benchmark-results.csv)를 참조하세요.

측정 native 실행파일 SHA256은 `0d3d56f7a8133b01d9bcf6ea8414a3e05bfb50ee248aaa8b8695a3db9f06a7bb`, metallib SHA256은 `7d8388c89c7a8bdac1cfc065b9daa1e1b02653cb5b7a78020cb9cb4bc0efa3e3`, server Python SHA256은 `746fbf6fc7f0a20a2f26dfa675516c2fd02e677640e6b94d642e84ea1df4f4c0`입니다. D는 검증된 library/metallib에 연결한 별도 벤치마크 adapter이므로 배포 main 실행파일과 hash가 다릅니다. Pilot4개 lane은 일반 실행 reference와 출력 토큰 및 최종 GDN/draft digest가 일치했습니다. 이 검증은 해당 범위에 적용되며 모든 state/KV byte의 보편적인 일치를 주장하지 않습니다.

공개 패키지는 검증된 native·Metal·서버·설치 코드와 의존성 버전을 유지합니다. 공개용 문서와 metadata를 정리하면서 archive hash는 달라질 수 있습니다. 패키지 설치·CLI·소스 계약·twine·payload hash 검증을 수행합니다.

기존60분 soak는 동일 payload의 지원32K/48G profile에 적용됩니다. 새로운96GiB/near256Ki 범위의 증거는 이번 전체 측정표입니다. 최종 설치물의10분 streaming·취소·재수용·RAM prefix 재사용·drain sanity는 별도 검사입니다. SSD는 opt-in이며 기존 동일 artifact의 제한된 restore 검증은 비공개 검토 증거에 연결합니다. Pi 영상은 SSD 가속 데모를 주장하지 않습니다.

Production 물리 폭은 B1–B4이며 B5–B8/cost-cap 연구는 수행하지 않았습니다. Logical262144 token에는 출력과 검증된 headroom이 포함됩니다. 요청 client 수·resident state 수·실제 dispatch 폭은 서로 구분합니다. Native Metal peak, sample 기반 process별 physical footprint, process-tree RSS는 합산할 수 없는 별도 메모리 지표입니다. 메모리 안내에는 각 실행 peak의 최대값을 사용합니다. 다른 RAM 구성, 자연 발생 critical-pressure SSD restore, 전원 손실 내구성은 인증하지 않습니다.


작업용 무추론 PTY 준비 검사기의 CPU 간섭과 겹친32Ki C4 D2개 실행을 동일 조건으로 재측정했습니다. 원본과 새 결과를 비공개 증거에 보존했습니다. 측정 전에 정한 규칙에 따라 처음 유효한 clean 실행을 사용했으며 fixture·설정·cap·n·통계를 바꾸거나 빠른 결과를 선택하지 않았습니다.

사용자가 보고한 한국시간 GPU 외부 작업 구간과 겹친128Ki 실행3개를 제외하고 동일 조건으로 재측정했습니다. 원본·시간대·겹친 구간·새 결과 hash를 비공개 증거에 보존했습니다. 컨트롤러 연결 중단으로 exit status를 확인할 수 없었던 실행도 별도 보존하고 동일 조건으로 반복했습니다.
