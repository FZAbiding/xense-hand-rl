# Wuji v8 diagnostic pilot: frozen checkpoint selection

Written during the 2026-10-17 sensor-only short pilot, before its final validation results. Only the fixed-validation seed `2026101701` may select a checkpoint. The unseen seed `2026101703` stays unopened until a model first passes the fixed gate and the contact and motion review is resolved.

Each saved validation is 24 complete, independent CPU episodes. Rank checkpoints lexicographically by:

1. More episodes passing **all** frozen v8 machine gates.
2. More episodes completing the full 5.5 seconds.
3. Lower mean fraction of frames with non-sensor force >0.05 N.
4. Higher mean dual-sensor contact fraction.
5. Earlier training iteration.

If no checkpoint passes the machine gate, label the selected checkpoint **diagnostic only**; do not promote it, use it as a qualified imitation target, or start a longer course. Preserve the final checkpoint, the selected diagnostic checkpoint, every evaluation's failures, and the original v7 baseline separately. Natural finger-pad contact and human-motion acceptance remain independent.
