"""Quest's Pinocchio/CasADi IK objective, without ROS or a viewer.

Based on agilexrobotics/questVR_ws teleop_double_piper.py at SOURCE_COMMIT.
The upstream task weights and 0.13 m tool transform are retained. Joint
regularization is relative to the last accepted command, not the zero pose.
Model assets remain in the separately downloaded upstream repository.
"""

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


class IKRejected(RuntimeError):
    pass


def portable_urdf(model_dir):
    """Upstream URDF embeds a developer's absolute mesh paths."""
    root = Path(model_dir).expanduser().resolve()
    tree = ET.parse(root / "urdf/piper_description.urdf")
    for mesh in tree.iter("mesh"):
        name = Path(mesh.attrib["filename"]).name
        path = root / "meshes" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", str(path))
    return ET.tostring(tree.getroot(), encoding="unicode")


class QuestIK:
    def __init__(self, model_dir, driver_limits):
        try:
            import casadi as ca
            import pinocchio as pin
            from pinocchio import casadi as cpin
        except ImportError as exc:
            raise RuntimeError("Quest IK needs Pinocchio with CasADi support; see docs/QUEST_COLLECTION.md") from exc
        self.pin = pin
        # Only the temporary rewritten URDF is needed; meshes use absolute paths.
        with tempfile.TemporaryDirectory(prefix="quest-model-") as tmp:
            urdf = Path(tmp) / "piper.urdf"
            urdf.write_text(portable_urdf(model_dir))
            self.full = pin.RobotWrapper.BuildFromURDF(str(urdf))
        self.robot = self.full.buildReducedRobot(["joint7", "joint8"], np.zeros(self.full.model.nq))
        self.model = self.robot.model
        if self.model.nq != 6:
            raise ValueError("Quest model must have six arm joints")
        # xyzrpy(0,0,0,0,-1.57,0) @ translation(.13,0,0)
        angle = -1.57
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        tool = pin.SE3(rotation, rotation @ np.array([.13, 0, 0]))
        self.model.addFrame(pin.Frame("ee", self.model.getJointId("joint6"), tool, pin.FrameType.OP_FRAME))
        self.frame = self.model.getFrameId("ee")
        self.data = self.model.createData()
        limits = np.asarray(driver_limits, dtype=float)
        if limits.shape != (6, 2) or not np.isfinite(limits).all():
            raise ValueError("invalid driver limits")
        self.limits = np.column_stack((np.maximum(limits[:, 0], self.model.lowerPositionLimit),
                                      np.minimum(limits[:, 1], self.model.upperPositionLimit)))
        if np.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError("driver and model limits do not overlap")
        self.geometry = self.full.collision_model
        # Upstream checks distal links/gripper against base/proximal links.
        if len(self.geometry.geometryObjects) < 9:
            raise ValueError("unexpected Quest collision geometry layout")
        for i in range(4, 9):
            for j in range(3):
                self.geometry.addCollisionPair(pin.CollisionPair(i, j))
        self.geom_data = pin.GeometryData(self.geometry)
        self.full_data = self.full.model.createData()
        symbolic = cpin.Model(self.model)
        data = symbolic.createData()
        q = ca.SX.sym("q", 6)
        target = ca.SX.sym("target", 4, 4)
        cpin.framesForwardKinematics(symbolic, data, q)
        error = ca.Function("quest_error", [q, target],
                            [cpin.log6(data.oMf[self.frame].inverse() * cpin.SE3(target)).vector])
        self.opti = ca.Opti()
        self.q = self.opti.variable(6)
        self.target = self.opti.parameter(4, 4)
        self.reference_q = self.opti.parameter(6)
        residual = error(self.q, self.target)
        self.opti.minimize(20 * (ca.sumsqr(residual[:3]) + ca.sumsqr(.1 * residual[3:]))
                           + .01 * ca.sumsqr(self.q - self.reference_q))
        self.opti.subject_to(self.opti.bounded(self.limits[:, 0], self.q, self.limits[:, 1]))
        self.opti.solver("ipopt", {"ipopt": {"max_iter": 50, "tol": 1e-7,
                                            "acceptable_tol": 1e-7,
                                            "print_level": 0, "sb": "yes"}, "print_time": False})
        self.seed = None

    def reset(self, measured):
        self.seed = np.asarray(measured, dtype=float).copy()

    def fk(self, joints):
        self.pin.framesForwardKinematics(self.model, self.data, np.asarray(joints, dtype=float))
        return self.data.oMf[self.frame].homogeneous.copy()

    def collides(self, joints, opening_m):
        full_q = np.concatenate((joints, [opening_m / 2, -opening_m / 2]))
        return bool(self.pin.computeCollisions(self.full.model, self.full_data, self.geometry,
                                              self.geom_data, full_q, False))

    def solve(self, target, opening_m):
        if self.seed is None:
            raise IKRejected("IK not anchored")
        self.opti.set_initial(self.q, self.seed)
        self.opti.set_value(self.reference_q, self.seed)
        self.opti.set_value(self.target, target)
        try:
            solution = self.opti.solve()
            q = np.asarray(solution.value(self.q)).reshape(6)
        except RuntimeError as exc:
            raise IKRejected("Quest IK did not converge") from exc
        if (not np.isfinite(q).all() or np.any(q < self.limits[:, 0] - 1e-6)
                or np.any(q > self.limits[:, 1] + 1e-6)):
            raise IKRejected("Quest IK output outside joint limits")
        # IPOPT can return a solution just beyond an active bound. Project only
        # its accepted numerical tolerance, then collision-check the exact q
        # that will pass the strict driver checks and be sent to hardware.
        q = np.clip(q, self.limits[:, 0], self.limits[:, 1])
        if self.collides(q, opening_m):
            raise IKRejected("Quest IK self collision")
        return q

    def accept(self, joints):
        # Commit warmstart only after the controller accepts/sends the solution.
        self.seed = np.asarray(joints, dtype=float).copy()
