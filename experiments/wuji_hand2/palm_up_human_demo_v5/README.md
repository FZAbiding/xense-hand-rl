# Wuji v5

已批准计划的标定与物理预演已完成，姿态等待人工审阅；没有开始训练。结果见 `artifacts/wuji_hand2/palm_up_human_demo_v5/review.html` 和 `reports/wuji_hand2/palm_up_human_demo_v5/REPORT.md`。

执行入口依次为 `prepare.py`、`verify_initial.py`、`settle_initial.py`、`search_motion.py --initial initial_settled --output motion_settled --seed 2026101207`、`probe_handover.py`、`review.py`。拒绝覆盖已有实验产物。
