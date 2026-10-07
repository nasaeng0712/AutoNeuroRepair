"""Human-readable TXT report for the actual-A01T basic integrity run.

Test-side utility only: validators return structured results and never write
files. Descriptions below are report text, not validation logic.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

REPORT_NAME = "Structural / Basic Integrity Test Report"

DESCRIPTIONS = {
    "DATA_ACCESS": ("신호 데이터를 Raw 객체에서 읽을 수 있는가",
                    "Signal data can be read from the Raw object"),
    "DATA_DIMENSION": ("데이터가 2차원 (채널, 샘플) 형태인가",
                       "Data is 2-D (channel, sample)"),
    "NON_EMPTY_DIMENSIONS": ("채널 수와 샘플 수가 모두 0보다 큰가",
                             "Channel and sample counts are both greater than 0"),
    "CHANNEL_SHAPE_CONSISTENCY": ("데이터 채널 수, 채널 이름 수, 채널 타입 수, 저장된 n_channels가 같은가",
                                  "Data channels, channel names, channel types and saved n_channels agree"),
    "SAMPLE_SHAPE_CONSISTENCY": ("데이터 샘플 수, raw.n_times, 저장된 n_times가 같은가",
                                 "Data samples, raw.n_times and saved n_times agree"),
    "SAMPLING_INFORMATION_VALIDITY": ("sfreq가 존재하고 숫자이며 유한하고 0보다 크며 저장값과 같은가",
                                      "sfreq exists, is numeric, finite, positive and equals the saved value"),
    "CHANNEL_METADATA_CONSISTENCY": ("채널 이름/타입 개수가 일치하고 빈 이름과 중복이 없으며 저장값과 같은가",
                                     "Channel name/type counts agree, no empty or duplicate names, equal to saved snapshot"),
    "FINITE_VALUES": ("전체 신호에 NaN, +Inf, -Inf가 없는가",
                      "No NaN, +Inf or -Inf in the whole signal"),
    "STRUCTURAL": ("Structural 검증 결과가 PASS인가", "Structural validation is PASS"),
    "FINITE": ("Finite 검증 결과가 PASS인가", "Finite validation is PASS"),
}


def git_state(repo):
    """Return (HEAD SHA, True if the working tree has uncommitted changes)."""
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))


def write_report(path, *, results, record, dataset_file, environment, commit):
    """results: output of validate_basic_integrity. commit: (sha, dirty)."""
    structural, finite, basic = results["structural"], results["finite"], results["basic_integrity"]
    lines = [
        f"{REPORT_NAME} / 구조 무결성 테스트 보고서",
        "=" * 70,
        f"report name / 보고서 이름: {REPORT_NAME}",
        f"validator version / 검증기 버전: structural={structural['validator_version']}, "
        f"finite={finite['validator_version']}, basic_integrity={basic['validator_version']}",
        f"timestamp / 시각 (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"tested commit SHA / 테스트 commit: {commit[0]}"
        + (" (working tree has uncommitted changes / 커밋되지 않은 변경 있음)" if commit[1] else ""),
        f"Python: {platform.python_version()}",
        f"MNE: {environment['mne']}",
        f"pytest: {environment['pytest']}",
        f"dataset/file / 데이터셋/파일: {record['dataset']} / {dataset_file}",
        f"SHA-256: {record['source_sha256']}",
        f"artifact_id: {record['artifact_id']}",
        f"processing_status (context only / 참고용, 검증 결과에 영향 없음): {record['processing_status']}",
        "",
    ]
    for result in (structural, finite, basic):
        lines += [f"[{result['validator_name']}] {result['status']}", "-" * 70]
        for item in result["checks"]:
            korean, english = DESCRIPTIONS[item["check_id"]]
            lines += [
                f"{item['check_id']}",
                f"  설명(KO): {korean}",
                f"  Description(EN): {english}",
                f"  criterion / 기준: {item['criterion']}",
                f"  observed / 관측값: {json.dumps(item['observed'], ensure_ascii=False)}",
                f"  result / 결과: {item['status']}",
                f"  failure reason / 실패 사유: {item['reason'] or '-'}",
            ]
        lines.append("")
    lines += [
        "=" * 70,
        f"Structural result / 구조 검증 결과: {structural['status']}",
        f"Finite result / 유한값 검증 결과: {finite['status']}",
        f"Basic Integrity result / 기본 무결성 결과: {basic['status']}",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
