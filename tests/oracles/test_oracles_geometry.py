# ==========================================================
# tests/oracles/test_oracles_geometry.py
# ==========================================================
# Hand-computed geometry oracles from spec E5.

import numpy as np

from lidar_sim.geometry.scene2d import Scene2D
from lidar_sim.world.scenes import SceneSpec


# ------------------------------------------
# test_o05_single_segment_hit
# ------------------------------------------
# O-05, E5: a ray along +x intersects the segment at x=2.
def test_o05_single_segment_hit():
    spec = SceneSpec("wall", np.array([[[2.0, -1.0], [2.0, 1.0]]]), [4], [9])
    hit = Scene2D.from_spec(spec).intersect(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([0.0]))

    assert hit.hit.tolist() == [True]
    np.testing.assert_allclose(hit.r, [2.0], rtol=1e-9, atol=0.0)
    np.testing.assert_allclose(hit.point, [[2.0, 0.0]], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(hit.normal, [[-1.0, 0.0]], rtol=0.0, atol=1e-12)
    assert hit.material_id.tolist() == [4]
    assert hit.object_id.tolist() == [9]
    assert hit.surface_id.tolist() == [0]


# ------------------------------------------
# test_o06_nearest_parallel_and_endpoint
# ------------------------------------------
# O-06, E5: nearest valid segment wins; parallel misses; segment endpoints count.
def test_o06_nearest_parallel_and_endpoint():
    segments = np.array([
        [[2.0, -1.0], [2.0, 1.0]],
        [[3.0, -1.0], [3.0, 1.0]],
        [[1.0, 0.5], [3.0, 0.5]],
        [[4.0, 0.0], [4.0, 1.0]],
    ])
    spec = SceneSpec("fixture", segments, [0, 0, 0, 0], [0, 1, 2, 3])
    nearest = Scene2D.from_spec(spec).intersect(
        np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([0.0])
    )
    assert nearest.hit.tolist() == [True]
    np.testing.assert_allclose(nearest.r, [2.0], rtol=0.0, atol=1e-12)
    assert nearest.surface_id.tolist() == [0]

    parallel_spec = SceneSpec("parallel", segments[2:3], [0], [2])
    parallel = Scene2D.from_spec(parallel_spec).intersect(
        np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([0.0])
    )
    assert parallel.hit.tolist() == [False]

    endpoint_spec = SceneSpec("endpoint", segments[3:4], [0], [3])
    endpoint = Scene2D.from_spec(endpoint_spec).intersect(
        np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([0.0])
    )
    # endpoint at (4,0) has segment parameter u=0.
    assert endpoint.hit.tolist() == [True]
    np.testing.assert_allclose(endpoint.r, [4.0], rtol=0.0, atol=1e-12)
