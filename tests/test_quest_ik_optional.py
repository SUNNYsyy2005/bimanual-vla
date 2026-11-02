"""Run in quest_collection after setup_quest_assets.py; no robot attached."""
import time

import numpy as np
import pytest

from bimanual_vla.collection.quest import DEFAULT_MODEL_DIR
from bimanual_vla.collection.quest.ik import IKRejected, QuestIK, portable_urdf

pytest.importorskip("pinocchio.casadi")
pytest.importorskip("casadi")


@pytest.fixture(scope="module")
def solver():
    if not (DEFAULT_MODEL_DIR / "meshes/link2.STL").exists():
        pytest.skip("download the pinned Quest model first")
    limits = np.deg2rad([[-150, 150], [0, 180], [-170, 0], [-100, 100], [-70, 70], [-180, 180]])
    return QuestIK(DEFAULT_MODEL_DIR, limits)


def test_real_official_solver_finite_bounded_and_warmstart_committed_only_on_accept(solver):
    q = np.array([0., .7, -1.1, 0., .4, 0.])
    solver.reset(q)
    pose = solver.fk(q)
    result = solver.solve(pose, .035)
    assert np.isfinite(result).all()
    assert not solver.collides(result, .035)
    assert np.all(result >= solver.limits[:, 0]) and np.all(result <= solver.limits[:, 1])
    np.testing.assert_array_equal(solver.seed, q)
    solver.accept(result)
    np.testing.assert_array_equal(solver.seed, result)
    # Task tracking and continuity are both represented in the objective.
    assert np.linalg.norm(solver.fk(result)[:3, 3] - pose[:3, 3]) < .03


@pytest.mark.parametrize("q", [
    [1.570657, .006370, -.112102, -.509182, .285449, .489094],
    [-1.297949, .044506, -.004974, .089413, .137462, .367601],
])
def test_measured_boundary_pose_does_not_pull_joints_toward_zero(solver, q):
    q = np.asarray(q)
    solver.reset(q)
    target = solver.fk(q)
    for _ in range(30):
        result = solver.solve(target, .036)
        assert np.max(np.abs(result - q)) < .01
        assert np.linalg.norm(solver.fk(result)[:3, 3] - target[:3, 3]) < .001
        solver.accept(result)


def test_real_solver_continuous_small_teleop_targets(solver):
    q = np.array([0., .7, -1.1, 0., .4, 0.])
    solver.reset(q)
    pose = solver.fk(q)
    previous = solver.solve(pose, .035)
    solver.accept(previous)
    durations = []
    for _ in range(30):
        pose[0, 3] += .0005
        started = time.monotonic()
        result = solver.solve(pose, .035)
        durations.append(time.monotonic() - started)
        assert np.max(np.abs(result - previous)) < .20
        assert not solver.collides(result, .035)
        solver.accept(result)
        previous = result
    print(f"Quest warm IK: median={np.median(durations)*1000:.2f}ms p95={np.percentile(durations,95)*1000:.2f}ms")


def test_model_mesh_paths_are_portable(solver):
    xml = portable_urdf(DEFAULT_MODEL_DIR)
    assert str(DEFAULT_MODEL_DIR / "meshes/link2.STL") in xml
    assert 'filename="/home/robot' not in xml
