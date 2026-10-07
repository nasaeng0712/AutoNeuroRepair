# Trial Assembly Metadata v1

구현: `src/anr/trial_assembly.py`. 실행 증거: `work/trial-assembly-evidence.txt`.
Event Semantic(`src/anr/event_semantics.py`)은 변경하지 않았고 COMPLETE 상태로 재사용한다.
이 단계는 metadata만 만든다. signal epoch 추출, tmin/tmax, Channel, Baseline은 범위 밖이다.

## 흐름

```text
load_a01t -> (raw, record)
raw.annotations (onset, description)
  -> source_sample_indices()          # onset -> 정수 sample index (exact)
  -> interpret_events()               # 기존 Event Semantic records
  -> require_semantic_complete()      # unknown_native_event 있으면 SemanticIncompleteError
  -> assemble_trials()                # trial metadata
  -> validate_a01t_trial_structure()  # A01T 규범 PASS/FAIL (assembly 규칙과 별개)
```

진입 함수는 `assemble_from_raw(raw, record)`이다. `semantic_complete` 같은 새 필드는 만들지 않았다.

## Source 위치

- `source_sample_index = round(onset * sfreq)`이며, `index / sfreq == onset`이 비트 단위로
  성립할 때만 받아들인다 (tolerance 없음). 범위는 `[0, n_samples)`.
- `raw.first_samp == 0`이고 `annotations.orig_time == raw.info["meas_date"]`일 때만
  onset을 source 시작 기준으로 본다. 아니면 `TrialAssemblyError`.
- 이벤트 identity는 annotation 배열 위치(`event_index`)이며 기존 Event Semantic record를
  `event` 필드에 그대로 복사해 둔다. 별도 event id 체계는 없다.

## 규칙

- 각 native `"768"`이 trial 하나다.
- trial span = `[anchor, min(다음 768, 다음 32766, source end))`, 시작 포함 / 끝 제외.
  "다음"은 sample index가 엄격히 큰 것이다. 소속은 sample index로만 정해지므로 같은 sample을
  공유하는 이벤트(예: 768과 1023)에 tie-break가 필요 없다. 같은 sample의 768 둘, 32766 둘,
  source 순서가 아닌 입력은 실패한다.
- span에는 task label(769~772)이 정확히 1개여야 한다 (0개/2개 이상은 실패). "다음 label 찾기"는 쓰지 않는다.
- cue timing: `label_sample - anchor_sample == 2 * sfreq` (A01T 250 Hz에서 500 samples).
- 모든 task label과 모든 1023은 정확히 하나의 span에 속해야 한다. orphan이면 실패한다.
- `trial_id = "<source_sha256>:<anchor_source_sample_index>"`. 식별 범위는
  `(dataset_id, task_id, trial_id)`이다. load UUID, 순번, list 위치는 쓰지 않는다.
  `source_artifact_id`는 lineage로만 보존한다.
- 1023은 span에 따라 trial에 연결하고 모든 lineage를 보존한다.
  `rejection_marker_present`는 기록일 뿐 trial 삭제나 Baseline 제외 결정이 아니다.
- run: 32766 sample로 timeline을 구분하고(`run_segment_index` = 해당 sample 이하의 32766 개수,
  0은 첫 32766 이전) 768 anchor를 포함한 segment만 task-bearing run으로 본다.
  32766 총수를 MI run 수로 해석하지 않는다.

## A01T 규범 기준 (`validate_a01t_trial_structure`)

anchor 288, 연결된 label 288, 4 label 각 72, task-bearing run 6, run당 trial 48,
run 내 label 각 12, cue 정확히 +2 s, trial id 유일, 모든 1023이 정확히 한 span에 귀속.
1023 관측 수(현재 15)는 증거로만 기록하며 상수로 요구하지 않는다.
