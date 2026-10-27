"""Physical execution limit for this rig, separate from dataset conventions.

2026-10-25: passive right-arm CAN feedback measured 91900 micrometres;
operator confirmed both grippers share the same limit. This is an execution
ceiling, not a calibration of the training dataset.
"""

GRIPPER_MAX_M = 0.0919
