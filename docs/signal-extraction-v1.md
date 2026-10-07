# Canonical Trial Signal Extraction

구현: `src/anr/signal_extraction.py` (`extract_trial_signals`, `window_n_times`).
실행 증거: `work/signal-extraction-evidence.txt`.
Event Semantic과 Trial Assembly는 변경하지 않았고 COMPLETE 상태로 재사용한다.

## 흐름

```text
load_a01t -> (raw, record) -> validate_basic_integrity
raw.annotations -> interpret_events -> require_semantic_complete
  -> assemble_from_raw(raw, record)            # Trial Assembly Metadata v1
  -> extract_trial_signals(raw.get_data(), assembly, record)
```

## 결정

- 768 anchor가 `t = 0`이다. cue(769~772)는 signal anchor가 아니며 Trial Assembly의 +2 s lineage로만 남는다.
- canonical window: `tmin = 0.0`, `tmax = 6.0`, `endpoint_convention = "half_open"`, 즉 `[0.0, 6.0)`.
- `n_times = (tmax - tmin) * sfreq`는 정확한 정수여야 한다. 아니면 `SignalExtractionError`이며
  round/floor/ceil을 하지 않는다. A01T는 250 Hz에서 1500 samples이다.
- 시작은 기존 `anchor.source_sample_index`(정수)를 그대로 쓰고 float onset으로 재구성하지 않는다.
  slice는 `source[:, start : start + n_times]`이고 마지막 포함 sample은 `start + 1499`이다.
- 추출 전에 모든 trial에서 `start >= 0`, `end_exclusive <= source_n_times`,
  `end_exclusive <= trial span end`를 검사한다. 하나라도 위반하면 collection 전체가 실패하며
  padding, crop, window 축소, silent drop은 없다.
- `rejection_marker_present = true`인 trial도 똑같이 추출한다. 삭제, skip, 값 변경은 없고
  Baseline 포함/제외 필드는 만들지 않았다 (Baseline 정책은 OPEN).
- 허용된 signal 연산은 sample 선택뿐이다. filtering, notch, resampling, normalization,
  rereferencing, interpolation, channel reorder/subset은 없다.
- `trial_id`는 Trial Assembly의 것을 그대로 쓴다 (새 ID 없음).
- epoch metadata: `trial_id`, `source_artifact_id`, `anchor_source_sample_index`,
  `signal_start_source_sample`, `signal_end_source_sample_exclusive`, `tmin`, `tmax`,
  `endpoint_convention`, `sfreq`, `n_times`, `ground_truth_semantic_label`
  (= Trial Assembly의 `semantic_label_id`), `rejection_marker_present`,
  `source_processing_status`(추출 시점의 `record["processing_status"]` 값).
- collection metadata는 provenance `record`를 재사용한다: source channel 이름/순서/타입, header 단위
  (`units`, `cal`)와 loader 변환 이력, source artifact id/SHA-256, processing status.
  `stage = "pre_channel_canonicalization"`.
- 출력 `[N, C_source, 1500]`은 최종 Canonical Trial Dataset이 아니다. 최종 `C`의 의미는 후속
  Channel Canonicalization이 정한다. 이번 단계는 source channel 표현을 그대로 유지한다.
