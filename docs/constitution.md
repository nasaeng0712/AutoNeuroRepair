## ANR Constitution v1.0 — PROJECT DECISION

**ANR의 목적**은 EEG/BCI 파이프라인에서 발생하는 신호 이상을 **Detection → Diagnosis → Repair → Automated Repair**로 처리하고, 최종적으로 **Self-Healing System**이 실제 downstream BCI 성능과 시스템 안정성을 유지·회복하는지 실험적으로 검증하는 것이다.

핵심 원칙은 다음 네 가지다.

| 원칙 | 확정 내용 |
|---|---|
| **Evidence First** | 코드가 동작한다는 주장은 automated test, 실험적 효과가 있다는 주장은 저장된 experiment result가 있어야 한다. |
| **Minimum Necessary Design** | 미래 단계의 Detection/Repair 구조를 지금 미리 상세 설계하지 않는다. 현재 단계에 필요한 만큼만 결정한다. |
| **Stable Architecture** | 새로운 아이디어, 다른 AI의 제안, 단일 실험 실패만으로 Architecture를 변경하지 않는다. |
| **Explainability of Decisions** | 중요한 설계는 사용자가 목적과 이유를 설명할 수 없는 상태에서 확정하지 않는다. |

추가로 ANR에서는 **“Self-Healing”이라는 이름 자체를 성공 주장으로 사용하지 않는다.** 마지막 단계까지 Evidence가 쌓이기 전까지는 연구 목표다.


<!-- Restored verbatim from Architecture & Control Tower; message 86dcafb6-721f-44ee-8f65-09137260dfc3. -->
