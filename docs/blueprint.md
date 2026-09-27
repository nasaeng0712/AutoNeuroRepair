# Blueprint v1.0 — PROJECT DECISION

장기 진행 순서는 그대로 확정한다.

**Pipeline  
→ Baseline / Initial Experiment  
→ Stabilization  
→ Condition / Parameter Expansion  
→ Experiment & Stabilization  
→ Detection  
→ Diagnosis  
→ Repair  
→ Automated Repair  
→ Self-Healing System**

여기서 중요한 것은 이게 **구현 목록이 아니라 Stage Gate**라는 점이다.

예를 들어 Detection으로 넘어가기 위해서는 단순히 “fault detector를 만들 수 있으니까”가 아니라, 그 전에 **정상 Pipeline과 Baseline이 안정적으로 재현되고, fault가 시스템에 어떤 영향을 주는지 실험적으로 확인되어 있어야 한다.**

따라서 지금부터는 앞 단계를 건너뛰어 미래 기능을 먼저 만드는 것을 기본적으로 BLOCK한다.


<!-- Restored verbatim from Architecture & Control Tower; message 86dcafb6-721f-44ee-8f65-09137260dfc3. -->
