"""Matrix FK and geometric Jacobian using the installed Piper SDK's modified DH.

No hardware access and no independent robot geometry or joint-limit defaults.
Keeping rotations as matrices avoids the SDK's intermediate Euler conversions.
"""

import numpy as np


class PiperKinematics:
    def __init__(self, sdk_fk):
        self.a = np.asarray(sdk_fk._a, dtype=np.float64) / 1000.0
        self.d = np.asarray(sdk_fk._d, dtype=np.float64) / 1000.0
        self.offset = np.asarray(sdk_fk._theta, dtype=np.float64).copy()
        self.ca = np.cos(sdk_fk._alpha)
        self.sa = np.sin(sdk_fk._alpha)
        if any(x.shape != (6,) or not np.isfinite(x).all()
               for x in (self.a, self.d, self.offset, self.ca, self.sa)):
            raise ValueError("Piper SDK DH parameters must be finite six-vectors")

    def evaluate(self, joints, *, jacobian=False):
        angles = np.asarray(joints) + self.offset
        c, s = np.cos(angles), np.sin(angles)
        links = np.zeros((6, 4, 4))
        links[:, 0, 0] = c
        links[:, 0, 1] = -s
        links[:, 0, 3] = self.a
        links[:, 1, 0] = s * self.ca
        links[:, 1, 1] = c * self.ca
        links[:, 1, 2] = -self.sa
        links[:, 1, 3] = -self.sa * self.d
        links[:, 2, 0] = s * self.sa
        links[:, 2, 1] = c * self.sa
        links[:, 2, 2] = self.ca
        links[:, 2, 3] = self.ca * self.d
        links[:, 3, 3] = 1.0
        transform = np.eye(4)
        origins, axes = np.empty((6, 3)), np.empty((6, 3))
        for i in range(6):
            transform = transform @ links[i]
            if jacobian:
                # Modified DH: joint i's axis is AFTER the fixed x transform.
                # Translation along that axis does not change its moment arm.
                origins[i] = transform[:3, 3]
                axes[i] = transform[:3, 2]
        xyz, rotation = transform[:3, 3].copy(), transform[:3, :3].copy()
        if not jacobian:
            return xyz, rotation
        geometric = np.concatenate((np.cross(axes, xyz - origins).T, axes.T), axis=0)
        return xyz, rotation, geometric


def rotation_error_jacobian(rotvec, angular_jacobian):
    """Derivative of Log(R_target R(q)^T) for spatial joint angular axes."""
    x, y, z = rotvec
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    theta = float(np.linalg.norm(rotvec))
    # Inverse right Jacobian of SO(3), stable at zero and near pi.
    coefficient = (1. / 12. + theta * theta / 720. if theta < 1e-4
                   else (1. - .5 * theta / np.tan(.5 * theta)) / (theta * theta))
    return -(np.eye(3) + .5 * skew + coefficient * (skew @ skew)) @ angular_jacobian
