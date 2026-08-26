# Global Arbiter A/B-only v1 Tracker

| Run ID | Stage | Purpose | New requests | Status |
|---|---|---|---:|---|
| GAB001 | `vlrb-global-arbiter-ab-only-freeze` | 冻结S0/S3/S4、S3 reports、Prompt和schedule | 0 | READY |
| GAB002 | `vlrb-global-arbiter-ab-only-audit` | 校验A/B-only parser、同replicate bundle和对照身份 | 0 | READY |
| GAB003 | `vlrb-global-arbiter-ab-only-smoke` | 20条、60请求端到端检查 | 60 | READY |
| GAB004 | `vlrb-global-arbiter-ab-only-run` | 1,247条全量重新仲裁 | 3,741 | READY |
| GAB005 | `vlrb-global-arbiter-ab-only-retry` | 对technical failures最多重试10次 | failures only | READY |
| GAB006 | `vlrb-global-arbiter-ab-only-report` | S5对S4/S3/S0、None子集和分层报告 | 0 | READY |

## Scope lock

- 只运行VL-RewardBench，不访问Discovery/Dev/heldout。
- 全部3,741个S5请求重新生成，不读取S4回答作为模型输入。
- S4只作为离线对照，S3五份root reports继续只读复用。
- 不按benchmark结果修改Prompt、Rubric、checkpoint或权重。
