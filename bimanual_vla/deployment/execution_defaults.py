"""Shared execution defaults for inference and recorded-episode replay."""
from bimanual_vla.deployment.servo import ServoConfig

_SERVO = ServoConfig()
DEFAULT_ACTION_HZ = 20.0
DEFAULT_SERVO_HZ = _SERVO.hz
DEFAULT_SPEED_PCT = 10
# Inference trajectory limits are configured in the GUI. Do not throttle the
# firmware independently through an invisible low-speed default.
DEFAULT_INFERENCE_SPEED_PCT = 100
DEFAULT_FEEDBACK_MAX_AGE_S = 0.5
DEFAULT_TRAJECTORY_MAX_SPEED_RAD_S = _SERVO.max_velocity
DEFAULT_TRAJECTORY_MAX_ACCELERATION_RAD_S2 = _SERVO.max_acceleration
DEFAULT_TRAJECTORY_MAX_JERK_RAD_S3 = _SERVO.max_jerk
DEFAULT_TRAJECTORY_MAX_TRACKING_ERROR_RAD = _SERVO.max_tracking_error
DEFAULT_GRIPPER_SPEED_M_S = _SERVO.max_gripper_velocity
