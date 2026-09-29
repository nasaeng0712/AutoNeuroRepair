# ANR-T001 v1.2 구현 및 검증 기록

기준은 사용자가 전달한 Implementation Specification v1.2와 이후의
"코드 작성과 실행 검증 분리" 지침이다. 이 문서는 새 Specification이 아니라
현재 코드의 동작과 미검증 상태를 설명한다.

## 현재 상태

- 코드 작성: 완료. T001 코드는 `main`에 commit되어 있다(`f775173`).
- Gate A: PASS. commit `f775173`, Python 3.11.5, pytest 8.4.2,
  `python -m pytest -m "not integration"` 결과 52 passed / 0 failed / 0 skipped.
  Gate A는 MNE를 import하지 않으며 실행 환경에 MNE는 설치되어 있지 않았다.
- Gate B: BLOCKED. 로컬 환경에 실제 A01T.gdf가 없다.
- T001 전체 PASS: 선언하지 않는다.
- T000-R의 승인 문서 복원은 이전 작업 결과다.

## 파일과 함수

| 파일 | 책임 |
|---|---|
| src/anr/loader.py | A01T.gdf 전용 MNE 로딩, 원본 식별, 메타데이터/변환 이력 수집 |
| src/anr/provenance.py | 근거와 이력 모델, 상태 판정, SHA-256, JSON 저장/재읽기/일관성 검사 |
| tests/test_provenance.py | 임시 바이트 파일과 Mock Reader를 쓰는 Gate A |
| tests/test_a01t_integration.py | 실제 파일을 요구하는 opt-in Gate B |
| pyproject.toml | MNE 1.11.0 의존성과 integration marker |
| README.md | 설치 및 Gate A/B 실행 명령 |

핵심 함수:

- `load_a01t()`는 `(raw, record)`를 반환한다. 기본 reader만 실제 MNE를 import한다.
- `_audit_loader()`는 MNE 1.11.0의 헤더 계약을 확인하고 변환 근거를 수집한다.
- `assess()`는 같은 단계/항목의 명시적인 값만 비교하며 세 상태를 판정한다.
- `file_sha256()`은 파일 바이트를 청크 단위로 읽는다.
- `save_record()`는 원본 덮어쓰기를 차단하고 새 JSON 파일만 생성한다.
- `read_record()`는 구조와 판정의 일관성을 확인한다. `source_path`를 지정하면
  실제 파일 SHA-256도 대조한다. JSON만 읽으면 파일 대조를 수행한 것으로 간주하지 않는다.

## 데이터 흐름

```text
A01T.gdf 경로 + 선택적 trusted SourceReference + 확인 근거/이력
  -> 파일명 및 입력 검사
  -> 원본 SHA-256 계산
  -> MNE read_raw_gdf (Gate A에서는 Mock Reader)
  -> 파일 SHA-256 재확인: 로딩 중 변경 시 오류
  -> 반환 메타데이터와 MNE 헤더 검사
  -> acquisition_history / processing_history / evidence 분리
  -> 상태, evidence_level, conflicts, review_required, unknowns 계산
  -> 로드별 UUID artifact_id와 source_sha256 연결
  -> Raw 객체와 JSON 가능한 record 반환
  -> JSON 저장 -> 재읽기 및 선택적 파일 바이트 대조
```

`artifact_id`는 로딩 결과별 UUID다. 같은 원본을 두 번 읽으면 source_sha256은
같고 artifact_id는 다르다. SHA-256을 artifact_id로 대신 사용하지 않는다.

## 판정 규칙과 근거의 한계

| 조건 | 결과 |
|---|---|
| 직접 확인된 추가 신호 처리 또는 확인된 처리 이력 있음 | PROCESSED |
| 위 조건과 동시에 명시적 충돌 있음 | PROCESSED 유지, conflicts 기록 및 review_required |
| 신뢰 기준과 파일 해시 일치 + Loader 검사 완료 + 미해결 충돌/이력 없음 | RAW |
| 출처 미확인, Loader 검사 미완료, 미확인 처리 주장 또는 미해결 충돌 | UNKNOWN |

RAW는 공식 배포 원본 대비 추가 신호 처리가 확인되지 않았다는 상대적 의미다.
처리 이력이 비어 있거나 파일명이 A01T.gdf라는 이유만으로 RAW를 부여하지 않는다.

`evidence`의 각 항목은 stage, item, value, source, kind를 가진다.
stage는 source/acquisition/loader/post_distribution,
kind는 document/metadata/direct/reported다. None은 정보 부재이며 충돌 값이 아니다.
숫자 50과 50.0은 같은 값으로 비교한다. 다른 단계의 같은 항목은 서로 충돌하지 않는다.
`evidence_level`의 DIRECT는 판정을 뒷받침하는 직접 근거가 있음을,
INSUFFICIENT는 상태 확정에 필요한 근거가 부족함을 나타낸다.
DIRECT는 과학적 확신도, 공식 인증 또는 Gate B PASS를 뜻하지 않는다.

원본 출처는 공식 자료 URL과 선택적 SourceReference에 남긴다. SourceReference는
사용자가 신뢰한 공식 취득 기록의 SHA-256, URL, 확인 설명이다. 프로그램이 그
설명의 진위를 인증하지는 않는다. 로컬 해시만 계산해 기준 해시로 다시 넣는 것은
출처 검증이 아니다. 신뢰 기준을 제공하지 않아도 로딩은 가능하며 상태는 보수적으로 UNKNOWN이다.

## 수집 이력과 Loader 변환

공식 수집 단계의 0.5-100 Hz 대역통과, 50 Hz 노치는 acquisition_history에
문서 근거로 저장한다. 코드가 필터를 실행했다거나 파형으로 측정했다고 표시하지 않는다.

MNE 옵션은 eog=[22,23,24], misc=None, stim_channel=None, exclude=[],
include=None, preload=True(기본), verbose=ERROR로 기록한다.
EOG 형 지정은 표현 메타데이터 설정이며 신호 처리가 아니다.

MNE 1.11.0 소스와 헤더의 cal/offsets/units 및 채널별 samples-per-record를
대조하여 calibration/offset/SI 변환과 암묵적 재표본화를 구분한다.
실제 MNE 버전과 사용한 옵션, 확인에 이용한 헤더 수치를 record에 남긴다.
preload=False에서는 아직 수행되지 않은 신호 디코딩을 확인된 변환으로 표시하지 않고
UNKNOWN을 반환한다. preload=True는 원본 디지털 스케일 보존 옵션이 아니다.

private `_raw_extras` 사용은 `_audit_loader()`에 한정된다. 이를 위해 확인한 버전
1.11.0을 고정했다. 버전/필수 필드가 다르면 RAW로 추정하지 않고 검사 미완료로 처리한다.
이 경로는 소스 대조만 했으며 실제 A01T.gdf와의 호환성은 Gate B에서 검증해야 한다.

raw.info의 필터 요약에는 기본값이 있을 수 있으므로 그 값만으로 공식 수집 기록과
충돌한다고 판단하지 않는다. 원본 헤더의 양수 필터 값만 명시적인 비교에 사용한다.
없거나 NaN인 헤더 값은 정보 부재로 기록한다. 0 또는 음수의 필터 값은 인코딩 의미가
확인되지 않았으므로 값은 보존하고 검토 필요로 처리한다. 이를 자동으로 "필터 없음"
또는 "정보 없음"으로 확정하지 않는다. Reader 경고도 보존하되 자동으로 충돌이라고
만들지 않으며 검사 미완료로 표시한다.

## 작성된 테스트

Gate A: 세 상태, 미확인/직접 확인 처리, 동일 단계/항목 충돌, 처리 이력과 부정 주장
충돌, 정보 부재, 반환 메타데이터 불일치, 단위/암묵적 재표본화 이력, 옵션과 버전,
lazy loading, Reader 경고와 예외, 파일 부재/권한/파일명/로딩 중 변경,
샘플 부재, UUID/해시 연결, JSON 왕복과 이동 파일 검증, JSON 오류/변조,
원본·기존 JSON 덮어쓰기 방지, 저장 경로·권한 오류를 포함한다.

Gate A의 임시 파일은 GDF 형식이 아니다. 결과에는 validation_scope=mock_reader,
mne_version=null이 기록된다. 테스트에서 RAW라는 결과를 얻더라도 실제 EEG 검증이 아니다.

Gate B: 실제 MNE 반환 객체와 공식 문서의 250 Hz, 25채널(22 EEG + 3 EOG),
288 trial 시작 및 네 클래스 각 72 cue를 대조한다. 출처 해시와 artifact 연결,
loader 옵션, 수집/처리 이력, JSON 왕복을 확인하고, 저장된 evidence와
processing_history로 판정을 다시 계산해 기록된 결과와 일치하는지 확인한다.
특정 processing_status(RAW)를 요구하지 않는다. RAW/PROCESSED/UNKNOWN 중
evidence와 일관된 결과이면 Software Verification을 통과하며, A01T의 실제 판정은
별도로 출력한다. 취득 기록(SourceReference)은 선택이며 없으면 UNKNOWN이 예상된다.
실제 파일 입력이 없으면 BLOCKED 메시지와 함께 skip한다.
전체 pytest 명령이 성공 종료해도 Gate B skip은 전체 T001 PASS가 아니다.

## 사용 예

```python
from pathlib import Path
from anr.loader import load_a01t
from anr.provenance import save_record, read_record

raw, record = load_a01t(r"C:\data\A01T.gdf")
try:
    # 출처 기준을 생략한 이 예의 예상 상태는 UNKNOWN이다.
    target = Path("results") / (record["artifact_id"] + ".json")
    save_record(record, target)
    restored = read_record(target, source_path=r"C:\data\A01T.gdf")
finally:
    raw.close()
```

results 디렉터리는 프로젝트 구조에 이미 존재한다. JSON에는 신호 배열/Raw 객체를
저장하지 않는다. 반환된 Raw 객체를 이후 수정해도 기록이 자동 갱신되지 않는다.
파일의 로딩 전후 해시 비교도 모든 동시 파일 변경을 방지하는 잠금은 아니다.
JSON 검사는 일관성 검사이며 전자서명/악의적인 동시 변조 감지 기능이 아니다.

## 확인한 자료

정적 검토 결과: 데이터 흐름과 예외 경로를 직접 읽고 점검했다. 확인된 추가 처리와
부정 주장의 충돌 보존, source reference와 JSON 근거의 연결 검사, 0 값 필터의
불명확한 의미를 보완했다. 이번 T001 변경 파일 7개에서 후행 공백은 발견되지 않았다.
pytest 수집 결과는 53개다(Gate A 52개, Gate B 1개). 실행 결과는 위 "현재 상태"에 기록했다.

- 공식 데이터 문서: https://www.bbci.de/competition/iv/desc_2a.pdf
- MNE 1.11.0 구현: https://github.com/mne-tools/mne-python/blob/v1.11.0/mne/io/edf/edf.py
- MNE 버전 요구사항: https://github.com/mne-tools/mne-python/blob/v1.11.0/pyproject.toml

MOABB, PSD, 배열 체크섬, Detection/Diagnosis/Repair는 도입하지 않았다.
