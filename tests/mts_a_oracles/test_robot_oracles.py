# ==========================================================
# tests/mts_a_oracles/test_robot_oracles.py
# ==========================================================
# Independent F7/F8/F19 oracles; final-spec v0.3 §§4, 7.4–7.5.
#
# Continuity is required on smooth segments, not at F8's hybrid speed jumps.
# The 60 pilot-seeded follower cases are the requested fast subset of T-R4's 200.

from itertools import pairwise

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mts.config import RobotConfig
from mts.robot import NavGrid, end_speed_bound, follow, safe_speed, sensor_pose, step
from mts.world import World, yard


# ------------------------------------------
# arena
# ------------------------------------------
# No walls; put the required target far outside this dynamics/navigation fixture.
def arena(*, boxes=(), rough=(), bounds=(0, 8, 0, 6)):
    return World(
        arena=bounds,
        box_lo=np.array([b[0] for b in boxes]).reshape(-1, 3),
        box_hi=np.array([b[1] for b in boxes]).reshape(-1, 3),
        box_material=np.zeros(len(boxes), dtype=int),
        rough=np.array(rough).reshape(-1, 4),
        prior_lo=np.array([100, 100, 1]), prior_hi=np.array([101, 101, 2]),
        target_half=np.full(3, 0.25), target_material=0,
        start=np.array([1., 1., 0., 0.]),
    )


# ------------------------------------------
# clearance
# ------------------------------------------
# Euclidean point/rectangle distance from raw fixture geometry (§4.2).
def clearance(world, cfg, points):
    lo = np.vstack((world.box_lo[:, :2], (world.prior_lo - world.target_half)[:2]))
    hi = np.vstack((world.box_hi[:, :2], (world.prior_hi + world.target_half)[:2]))
    lo = lo - cfg.radius - cfg.margin
    hi = hi + cfg.radius + cfg.margin
    delta = np.maximum(np.maximum(lo - np.asarray(points)[..., None, :],
                                  np.asarray(points)[..., None, :] - hi), 0)
    return np.linalg.norm(delta, axis=-1).min(axis=-1)


# ------------------------------------------
# smooth_solution
# ------------------------------------------
# Integrals of (v0+a*t)*cos/sin(theta0+omega*t), no integrator calls.
def smooth_solution(x, a, omega, t):
    px, py, theta, v = x
    angle = theta + omega * t
    if omega == 0:
        distance = v * t + a * t**2 / 2
        dx, dy = distance * np.cos(theta), distance * np.sin(theta)
    else:
        dx = ((v + a*t)*np.sin(angle) - v*np.sin(theta))/omega
        dx += a*(np.cos(angle) - np.cos(theta))/omega**2
        dy = (v*np.cos(theta) - (v + a*t)*np.cos(angle))/omega
        dy += a*(np.sin(angle) - np.sin(theta))/omega**2
    return np.array([px + dx, py + dy, angle, v + a*t])


# ------------------------------------------
# test_tr1_straight_acceleration
# ------------------------------------------
# Straight-line motion under constant acceleration matches the closed form.
@pytest.mark.parametrize('v0', [0., 0.25])
@pytest.mark.parametrize('theta', [0., 0.7, -1.2])
def test_tr1_straight_acceleration(v0, theta):
    cfg = RobotConfig()
    x = np.array([2., 2., theta, v0])
    tr = step(arena(), cfg, x, [0.6, 0])
    expected = [smooth_solution(x, 0.6, 0, t) for t in np.arange(26)*0.02]
    assert_allclose(tr.states, expected, rtol=0, atol=2e-14)


# ------------------------------------------
# test_tr1_constant_speed_circle
# ------------------------------------------
# Constant-speed turning matches the closed-form circle.
@pytest.mark.parametrize('omega', [-0.8, 0.8])
def test_tr1_constant_speed_circle(omega):
    x = np.array([2., 2., 0.4, 0.65])
    tr = step(arena(), RobotConfig(), x, [0, omega])
    for i, state in enumerate(tr.states):
        assert_allclose(state, smooth_solution(x, 0, omega, i*0.02), atol=2e-11, rtol=0)
    centre = x[:2] + x[3]/omega * np.array([-np.sin(x[2]), np.cos(x[2])])
    assert_allclose(np.linalg.norm(tr.states[:, :2] - centre, axis=1), abs(x[3]/omega), atol=2e-11)


# ------------------------------------------
# test_tr1_rk4_fourth_order
# ------------------------------------------
# The integrator's error falls with the fourth power of the step.
def test_tr1_rk4_fourth_order():
    x = np.array([2., 2., 0.3, 0.2])
    exact = smooth_solution(x, 0.7, 0.9, 0.5)
    errors = []
    for h in [0.1, 0.05, 0.02, 0.01]:
        tr = step(arena(), RobotConfig(substep=h), x, [0.7, 0.9])
        errors.append(np.linalg.norm(tr.end[:2] - exact[:2]))
    assert errors[2] < 1e-9  # configured 20 ms substep versus closed form
    assert 12 < errors[0]/errors[1] < 20
    assert 12 < errors[2]/errors[3] < 20


# ------------------------------------------
# test_tr2_control_clipping
# ------------------------------------------
# Controls beyond the limits are clipped.
@pytest.mark.parametrize('u', [[20, 30], [-20, -30]])
def test_tr2_control_clipping(u):
    cfg = RobotConfig()
    x = np.array([2., 2., 0., 0.5])
    tr = step(arena(), cfg, x, u)
    a, omega = np.sign(u)
    assert_allclose(tr.u, [a, omega])
    assert_allclose(tr.end, smooth_solution(x, a, omega, cfg.dt), atol=2e-10, rtol=0)


# ------------------------------------------
# test_tr2_speed_limits
# ------------------------------------------
# Speed stays between zero and the limit.
@pytest.mark.parametrize(('v', 'a', 'bound'), [(0.9, 10, 1.), (0.1, -10, 0.)])
def test_tr2_speed_limits(v, a, bound):
    cfg = RobotConfig()
    tr = step(arena(), cfg, [2, 2, 0, v], [a, 0])
    assert np.all((tr.states[:, 3] >= 0) & (tr.states[:, 3] <= 1))
    assert tr.end[3] == pytest.approx(bound)


# ------------------------------------------
# test_tr2_rough_entry_projection_and_exit_acceleration
# ------------------------------------------
# Entering rough ground caps the speed; leaving it lets the robot accelerate again (F8).
def test_tr2_rough_entry_projection_and_exit_acceleration():
    cfg = RobotConfig()
    world = arena(rough=[(2, 2.08, 1, 3)])
    tr = step(world, cfg, [1.99, 2, 0, 0.8], [1, 0])
    # First free-space RK4 substep travels .8*.02 + .02**2/2, then F8 projects.
    assert_allclose(tr.states[1], [2.0062, 2, 0, 0.3], atol=1e-14)
    inside = (tr.states[:, 0] >= 2) & (tr.states[:, 0] <= 2.08)
    assert inside.any()
    assert_allclose(tr.states[inside, 3], 0.3, atol=1e-14)
    exit_index = np.flatnonzero(tr.states[:, 0] > 2.08)[0]
    assert tr.states[exit_index, 3] == pytest.approx(0.32)
    assert tr.end[3] > 0.3


# ------------------------------------------
# assert_pose
# ------------------------------------------
# Checks a sensor pose against the robot state (F1).
def assert_pose(pose, state, height):
    c, s = np.cos(state[2]), np.sin(state[2])
    assert_allclose(pose.t, [state[0], state[1], height], atol=1e-12)
    assert_allclose(pose.R, [[c, -s, 0], [s, c, 0], [0, 0, 1]], atol=1e-12)


# ------------------------------------------
# test_tr2_trajectory_interpolation_and_f1_pose
# ------------------------------------------
# Trajectory interpolation and the sensor pose at any time (F1).
def test_tr2_trajectory_interpolation_and_f1_pose():
    cfg = RobotConfig(h_s=0.63)
    x = np.array([2., 2., -0.4, 0.25])
    tr = step(arena(), cfg, x, [0.4, 0.7], t0=3.25)
    assert_allclose(tr.states[0], x, atol=0)
    for i, state in enumerate(tr.states):
        t = tr.t0 + i*cfg.substep
        assert_allclose(tr.state(t), state, atol=2e-14, rtol=0)
        if 0 < i < cfg.n_sub:
            assert_allclose(tr.state(t-1e-9), tr.state(t+1e-9), atol=3e-9, rtol=0)
    for offset in [0.007, 0.133, 0.499]:
        expected = smooth_solution(x, 0.4, 0.7, offset)
        assert_allclose(tr.state(tr.t0 + offset), expected, atol=2e-11, rtol=0)
        assert_pose(tr.pose(tr.t0 + offset), expected, cfg.h_s)
    assert_pose(sensor_pose(x, cfg), x, cfg.h_s)
    for t in [tr.t0-0.001, tr.t0+cfg.dt+0.001]:
        with pytest.raises(ValueError):
            tr.state(t)
        with pytest.raises(ValueError):
            tr.pose(t)


# ------------------------------------------
# test_tr2_step_keepout_feasibility
# ------------------------------------------
# A step that enters a keep-out zone is marked infeasible.
@pytest.mark.parametrize(('y', 'expected'), [(2., False), (3., True)])
def test_tr2_step_keepout_feasibility(y, expected):
    # Inflated obstacle is [1.85, 2.75] x [1.45, 2.55].
    world = arena(boxes=[((2.2, 1.8, 0), (2.4, 2.2, 1))])
    tr = step(world, RobotConfig(), [1.6, y, 0, 1], [0, 0])
    assert bool(tr.feasible) is expected
    assert bool(np.all(clearance(world, RobotConfig(), tr.states[:, :2]) > 0)) is expected


# ------------------------------------------
# test_tr4_end_speed_positive_root_and_brute_force
# ------------------------------------------
# The end-speed bound matches its positive root and a brute-force search.
@pytest.mark.parametrize(('v', 'd'), [(0., 0.2), (0.3, 0.4), (0.9, 2.), (0.6, 0.01), (0., 0.), (0.8, -1.)])
def test_tr4_end_speed_positive_root_and_brute_force(v, d):
    cfg = RobotConfig(a_max=0.8)
    actual = end_speed_bound(cfg, v, d)
    if d <= v*cfg.dt/2:
        assert actual == 0
        return
    # Quadratic formula with independently assembled polynomial coefficients.
    aa, bb, cc = 1/(2*cfg.a_max), cfg.dt/2, v*cfg.dt/2-d
    expected = (-bb + np.sqrt(bb*bb - 4*aa*cc))/(2*aa)
    assert actual == pytest.approx(expected, abs=1e-13)
    assert aa*actual**2 + bb*actual + cc == pytest.approx(0, abs=1e-13)
    scan = np.linspace(0, 5, 100_001)
    feasible = scan[aa*scan**2 + bb*scan + cc <= 0]
    assert 0 <= actual-feasible[-1] <= 5e-5 + 1e-13


# ------------------------------------------
# test_tr4_safe_cruise_quadratic
# ------------------------------------------
# The safe cruising speed solves the braking quadratic.
@pytest.mark.parametrize('c', [-1., 0., 0.05, 0.06, 0.4, 3.])
def test_tr4_safe_cruise_quadratic(c):
    cfg = RobotConfig(a_max=0.7)
    room = max(c-cfg.brake_margin, 0)
    result = float(safe_speed(cfg, c))
    expected = -cfg.a_max*cfg.dt + np.sqrt((cfg.a_max*cfg.dt)**2 + 2*cfg.a_max*room)
    assert result >= 0
    assert result == pytest.approx(expected, abs=1e-13)
    assert result*cfg.dt + result**2/(2*cfg.a_max) == pytest.approx(room, abs=1e-13)


# ------------------------------------------
# assert_grid_path
# ------------------------------------------
# Checks that a grid path runs between the right cells through free, adjacent cells.
def assert_grid_path(grid, pred, start, goal):
    path = grid.path(pred, goal)
    assert_allclose(path[0], grid.centre(start))
    assert_allclose(path[-1], grid.centre(goal))
    cells = [grid.cell_of(p) for p in path]
    for cell in cells:
        assert grid.free[cell]
    for (y, x), (yy, xx) in pairwise(cells):
        assert max(abs(yy-y), abs(xx-x)) == 1
        if yy != y and xx != x:
            assert grid.free[y, xx] and grid.free[yy, x], 'diagonal corner cutting'
    return path


# ------------------------------------------
# test_tr4_hand_computed_straight_run
# ------------------------------------------
# Travel time of a straight run, on smooth and rough ground, by hand.
@pytest.mark.parametrize('rough', [False, True])
def test_tr4_hand_computed_straight_run(rough):
    # One obstacle far from this row; every endpoint has c >= 1.05, hence cruise=1.
    world = arena(boxes=[((6, 4, 0), (7, 5, 1))],
                  rough=[(0, 8, 0, 6)] if rough else [])
    grid = NavGrid(world, RobotConfig(), cell=0.5)
    start, goal = grid.cell_of([1.25, 1.25]), grid.cell_of([4.25, 1.25])
    times, pred = grid.travel_time(grid.centre(start))
    assert times[goal] == pytest.approx(3/(0.3 if rough else 1), abs=1e-12)
    assert_grid_path(grid, pred, start, goal)


# ------------------------------------------
# test_tr4_rough_patch_detour_beats_direct_run
# ------------------------------------------
# A detour around rough ground beats driving through it.
def test_tr4_rough_patch_detour_beats_direct_run():
    world = arena(boxes=[((6, 4, 0), (7, 5, 1))], rough=[(2, 4, 2, 2.5)])
    grid = NavGrid(world, RobotConfig(), cell=0.5)
    start, goal = grid.cell_of([1.25, 2.25]), grid.cell_of([4.75, 2.25])
    times, pred = grid.travel_time(grid.centre(start))
    # Seven horizontal edges. Detour replaces two with diagonals, all at speed 1.
    detour_time = 5*0.5 + 2*np.sqrt(0.5**2 + 0.5**2)
    # Direct: five edges touch a rough centre; two do not.
    direct_time = 5*0.5/0.3 + 2*0.5
    assert times[goal] == pytest.approx(detour_time, abs=1e-12)
    assert times[goal] < direct_time
    path = assert_grid_path(grid, pred, start, goal)
    assert not np.any((path[:, 0] >= 2) & (path[:, 0] <= 4) &
                      (path[:, 1] >= 2) & (path[:, 1] <= 2.5))


# ------------------------------------------
# test_tr4_unreachable_across_barrier
# ------------------------------------------
# Cells behind a wall are unreachable.
def test_tr4_unreachable_across_barrier():
    world = arena(boxes=[((3.5, 0, 0), (4, 6, 1))])
    grid = NavGrid(world, RobotConfig(), cell=0.5)
    times, _ = grid.travel_time([1.25, 2.25])
    assert np.isinf(times[grid.cell_of([6.25, 2.25])])
    assert np.all(np.isinf(times[~grid.free]))


# ------------------------------------------
# test_tr4_corner_cutting_forbidden
# ------------------------------------------
# Paths never cut the corner of an obstacle.
def test_tr4_corner_cutting_forbidden():
    # Only diagonally adjacent opposite corners remain free in this 2x2 arena.
    cfg = RobotConfig(radius=0, margin=0, brake_margin=0)
    world = arena(bounds=(0, 2, 0, 2), boxes=[
        ((1.4, 0.4, 0), (1.6, 0.6, 1)), ((0.4, 1.4, 0), (0.6, 1.6, 1)),
    ])
    grid = NavGrid(world, cfg, cell=1)
    assert_allclose(grid.free, [[True, False], [False, True]])
    times, _ = grid.travel_time([0.5, 0.5])
    assert np.isinf(times[1, 1])


# ------------------------------------------
# test_tr4_cell_speed_and_free_rule_from_geometry
# ------------------------------------------
# Cell speeds and free cells follow from the geometry.
def test_tr4_cell_speed_and_free_rule_from_geometry():
    cfg = RobotConfig()
    world = arena(boxes=[((3, 2, 0), (4, 3, 1))], rough=[(1, 2, 1, 2)])
    grid = NavGrid(world, cfg, cell=0.25)
    xx, yy = np.meshgrid(grid.xs, grid.ys)
    points = np.stack([xx, yy], axis=-1)
    c = clearance(world, cfg, points)
    room = np.maximum(c-cfg.brake_margin, 0)
    cruise = np.sqrt((cfg.a_max*cfg.dt)**2 + 2*cfg.a_max*room)-cfg.a_max*cfg.dt
    limit = np.where((xx >= 1) & (xx <= 2) & (yy >= 1) & (yy <= 2), 0.3, 1.)
    expected = np.minimum(cruise, limit)
    free = (c > 0) & (expected >= 0.15)
    assert_allclose(grid.free, free)
    assert_allclose(grid.speed[free], expected[free], atol=1e-13)


# ------------------------------------------
# yard_grid
# ------------------------------------------
# The default yard, robot and navigation grid.
@pytest.fixture(scope='module')
def yard_grid():
    world, cfg = yard(), RobotConfig()
    return world, cfg, NavGrid(world, cfg)


# ------------------------------------------
# test_tr4_seeded_follow_arrival_safety_no_fallback
# ------------------------------------------
# Following random paths in the yard arrives safely without a fallback.
@pytest.mark.parametrize('case', range(60))
def test_tr4_seeded_follow_arrival_safety_no_fallback(yard_grid, case):
    world, cfg, grid = yard_grid
    rng = np.random.default_rng([94240, case])
    cells = np.argwhere(grid.free)
    start, goal = (tuple(cells[i]) for i in rng.choice(len(cells), size=2, replace=False))
    target = grid.centre(goal)
    x = np.r_[grid.centre(start), rng.uniform(-np.pi, np.pi), 0.]
    initial_times, _ = grid.travel_time(x[:2])
    budget = 4*initial_times[goal] + 10
    context = f'case={case}, start={start}, goal={goal}, initial={x.tolist()}'
    for k in range(int(np.ceil(budget/cfg.dt))):
        times, pred = grid.travel_time(x[:2])
        assert np.isfinite(times[goal]), context
        path = np.vstack([x[:2], grid.path(pred, goal)])
        u, fallback = follow(world, cfg, x, path, face=target)
        assert not fallback, context
        tr = step(world, cfg, x, u)
        assert tr.feasible, f'{context}, k={k}, x={x}, u={u}'
        # Check every visited substep, using independent rectangle distances.
        room = np.maximum(clearance(world, cfg, tr.states[:, :2])-cfg.brake_margin, 0)
        assert np.all(tr.states[:, 3]**2/(2*cfg.a_max) <= room+1e-9), context
        x = tr.end
        if np.linalg.norm(x[:2]-target) <= cfg.goal_tol and x[3] <= 1e-6:
            break
    else:
        pytest.fail(f'{context}: did not arrive in {budget:.3f}s; final={x.tolist()}')


# ------------------------------------------
# test_tr4_mixed_speed_corridor_uses_slower_edge_endpoint
# ------------------------------------------
# An edge between smooth and rough ground uses the slower speed.
def test_tr4_mixed_speed_corridor_uses_slower_edge_endpoint():
    # A single row forces a straight path. Obstacle bottom after inflation is y=1.15;
    # centre y=.5 gives c=.65, room=.60, cruise=sqrt(1.45)-.5.
    world = arena(bounds=(0, 5, 0, 1), boxes=[((-1, 1.5, 0), (6, 2, 1))],
                  rough=[(2, 3, 0, 1)])
    grid = NavGrid(world, RobotConfig(), cell=1)
    times, pred = grid.travel_time([0.5, 0.5])
    cruise = np.sqrt(1.45)-0.5
    # Two edges touch the lone rough cell; the other two use the clearance speed.
    assert times[0, 4] == pytest.approx(2/0.3 + 2/cruise, abs=1e-12)
    assert_grid_path(grid, pred, (0, 0), (0, 4))


# ------------------------------------------
# test_tr4_follow_control_matches_f19_by_hand
# ------------------------------------------
# The path follower's control matches F19 worked by hand.
@pytest.mark.parametrize('theta', [0.3, -0.6, 2.0])
def test_tr4_follow_control_matches_f19_by_hand(theta):
    cfg = RobotConfig()
    x = np.array([2., 2., theta, 0.2])
    # Goal is .5 m ahead; terrain and clearance do not constrain the speed.
    # v'^2/2 + v'/4 + .05 - .5 = 0 gives v' = (sqrt(3.85)-.5)/2.
    bound = (np.sqrt(3.85)-0.5)/2
    expected_a = np.clip((bound*max(np.cos(theta), 0)-0.2)/0.5, -1, 1)
    expected_w = np.clip(-2*theta, -1, 1)
    u, fallback = follow(arena(), cfg, x, np.array([[2., 2.], [2.5, 2.]]))
    assert_allclose(u, [expected_a, expected_w], atol=1e-13)
    assert not fallback


# ------------------------------------------
# test_tr4_follow_arrival_brakes_and_faces_requested_point
# ------------------------------------------
# On arrival the follower brakes and turns to face the requested point.
def test_tr4_follow_arrival_brakes_and_faces_requested_point():
    u, fallback = follow(arena(), RobotConfig(), [2, 2, 0, 0.2],
                         np.array([[2., 2.], [2.1, 2.]]), face=np.array([2., 3.]))
    assert_allclose(u, [-0.4, 1.], atol=1e-13)
    assert not fallback
