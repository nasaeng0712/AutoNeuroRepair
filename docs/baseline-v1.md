# CSP + LDA Baseline v1

구현: `src/anr/baseline.py`(모델/분할/전처리/평가), `src/anr/baseline_run.py`(최종 학습·평가·Freeze).
범위는 BCIC IV 2a A01T, subject-specific 4-class 운동상상이다. 범용 model framework가 아니다.
이후 Fault Injection, Detection, Diagnosis, Repair는 구현하지 않았다.

## 구성

- FrozenModelVocabulary(명시 순서): `0 left_hand(769)`, `1 right_hand(770)`, `2 both_feet(771)`, `3 tongue(772)`.
  Pipeline의 semantic label을 이 표로 정수로 바꾼다. estimator가 문자열을 정렬해 순서를 정하지 않는다.
- Split: Trial Assembly의 task-bearing run 1-4 = TRAIN, 5 = VALIDATION, 6 = TEST (무작위 없음).
  거부 마커(1023) trial은 Canonical Dataset에 남기고, 분할 manifest에
  `model_eligible=false`와 사유를 기록하며 fit과 평가 metric에서 제외한다.
  clean partition마다 4개 class가 모두 있어야 하며 없으면 `BaselineBlockedError`(Control Tower 확인).
- 입력: `X_eeg`의 22개 canonical EEG 채널 전부. EOG는 사용하지 않는다.
- 전처리(Baseline 전용): canonical `[0, 6)` EEG -> 8-30 Hz FIR band-pass
  (MNE `filter_data`, zero-phase, hamming, firwin, pad `reflect_limited`) -> crop `[2, 6)` = 1000 samples.
  notch, resampling, rereference, scaler는 없다. MNE가 실제로 정한 값(필터 길이 413 samples,
  전이대역 2.0/7.5 Hz)을 manifest에 기록한다.
- CSP: `mne.decoding.CSP(n_components=4, reg=None, cov_est="concat", transform_into="average_power",
  log=True, norm_trace=False, component_order="mutual_info")`.
- LDA: `LinearDiscriminantAnalysis(solver="svd", shrinkage=None, priors=None, n_components=None,
  store_covariance=False, tol=1e-4, covariance_estimator=None)`. calibration 없음.
- CSP와 LDA는 TRAIN clean trial에서 한 번만 fit한다. validation/test는 transform/predict만 한다.
- 예측: `trial_id, model_artifact_id, frozen_class_order, predicted_model_class_index,
  predicted_semantic_label, probability_vector, decision_score_vector`.
  `probability_semantics = "native_lda_model_estimate"`(보정된 신뢰도가 아님),
  `decision_score_semantics = "lda_decision_function"`.
- `model_artifact_id = "<source_sha256>:csp-lda-baseline-v1"` (dataset id와 같은 규칙).

## 판정과 Freeze

- Software Verification(누수 없음, fit 경계, class/확률/decision 불변식, reload 재현성)과
  Functional Qualification(validation/test 모두 `balanced_accuracy > 0.25`이고 `kappa > 0`)을 분리한다.
  0.25는 4-class 명목 chance 기준일 뿐 통계적 유의성이나 SOTA 주장이 아니다.
- 둘 다 통과한 exact fitted stack(`model.joblib`: 학습된 CSP와 LDA 객체 전체)만 Freeze한다.
  실패하면 `Frozen Baseline = NOT APPROVED`로 결과만 저장하고 tuning하지 않는다.
- Freeze Manifest의 `training_code_commit_sha`는 학습을 실제로 수행한 clean commit이다.
  `baseline_run`은 working tree가 dirty이면 실행을 거부한다.
- Frozen inference boundary: 이후 비교(reference/corrupted/repaired)에서 filter, window, channel set,
  CSP/LDA 상태를 바꾸지 않는다. scaler와 calibration은 없고 online adaptation도 없다.
