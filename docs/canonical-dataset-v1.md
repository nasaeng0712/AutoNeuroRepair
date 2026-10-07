# Channel Canonicalization and Canonical Trial Dataset v1

구현: `src/anr/channel_canonicalization.py`, `src/anr/canonical_dataset.py`,
`src/anr/pipeline.py`(기존 단계를 순서대로 호출하는 얇은 orchestration).
실행 증거: `work/canonical-dataset-evidence.txt`.

## Channel Canonicalization (channel_schema_v1)

- `[N, 25, T]`를 `X_eeg [N, 22, T]`와 `X_eog [N, 3, T]`로 나눈다. EOG는 버리지 않는다.
- canonical id는 dataset 범위의 순번이다: `bciciv2a:eeg:01..22`, `bciciv2a:eog:01..03`.
  10-20 전극 이름을 복원하지 않는다. 각 id는 loader가 A01T에서 실제로 내놓은 source channel
  이름과 순서에 묶여 있다 (`EEG-Fz, EEG-0, ..., EEG-Pz, EEG-15, EEG-16`, `EOG-left/central/right`).
- 허용 연산은 EEG/EOG 그룹으로의 channel 선택과 결정적 reorder뿐이다. 값은 수정하지 않는다.
- 다음은 모두 `ChannelCanonicalizationError`로 실패한다: 필수 channel 누락, 중복 identity,
  모호한 schema, type 불일치, 추가 channel, 검증되지 않은 alias(이름이 정확히 같아야 함).

## Canonical Trial Dataset

- 구성: `X_eeg`, `X_eog`, `y_semantic`(Pipeline 의미 label 문자열), `trial_metadata`, `dataset_metadata`.
- `canonical_dataset_id = "<source_sha256>:canonical-trial-dataset-v1"`.
- `trial_id`는 Trial Assembly의 것을 그대로 쓴다. `model_class_index`, filtering, crop, feature는 없다.
- `source_processing_status`는 생성 시점 provenance의 snapshot이며 dataset 자체의 RAW/PROCESSED
  판정이 아니다. 이 모듈은 provenance를 다시 판정하지 않는다.
- `pipeline_transformation_history`에는 실제 수행한 것만 기록한다: `[0, 6)` source sample 선택,
  EEG/EOG channel canonicalization. Baseline 전용 처리(8-30 Hz, `[2, 6)`, CSP)는 포함하지 않는다.
- trial 메타데이터에는 `task_run_index`(task-bearing run의 1부터의 순번)와 거부 마커 lineage가 있다.
- 불변식은 `validate_canonical_dataset`가 PASS/FAIL로 검사한다 (양수 차원, finite, 개수 일치,
  trial id, 공통 epoch/sfreq/lineage, channel schema, label, model 필드 부재).
