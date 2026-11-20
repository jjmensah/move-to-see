# ==========================================================
# tests/unit/test_types.py
# ==========================================================
# Unit tests of the record types.

import numpy as np
import pytest

from lidar_sim.core.pose import Pose
from lidar_sim.core.types import (
    BeamPrediction,
    HitBatch,
    MLEResult,
    RadiometricScan,
    Scan,
)


# ------------------------------------------
# _hits
# ------------------------------------------
# A valid batch of hits.
def _hits(n=4, d=2):
    return HitBatch(hit=np.ones(n, bool), r=np.ones(n), point=np.zeros((n, d)),
                    normal=np.tile(np.eye(d)[0], (n, 1)), material_id=np.zeros(n, int),
                    object_id=np.zeros(n, int), surface_id=np.zeros(n, int))


# ------------------------------------------
# test_hitbatch_valid_and_readonly
# ------------------------------------------
# A valid hit batch is read-only.
def test_hitbatch_valid_and_readonly():
    h = _hits()
    assert h.n == 4 and h.dim == 2
    with pytest.raises(ValueError):
        h.r[0] = 2.0                         # arrays are frozen (read-only)


# ------------------------------------------
# test_hitbatch_shape_error_names_field
# ------------------------------------------
# A shape error names the offending field.
def test_hitbatch_shape_error_names_field():
    with pytest.raises(ValueError, match="normal"):
        HitBatch(hit=np.ones(3, bool), r=np.ones(3), point=np.zeros((3, 2)),
                 normal=np.zeros((3, 3)), material_id=np.zeros(3, int),
                 object_id=np.zeros(3, int), surface_id=np.zeros(3, int))


# ------------------------------------------
# test_radiometric_scan_empty_valid
# ------------------------------------------
# An empty radiometric scan is valid in 2D and 3D.
@pytest.mark.parametrize("d", [2, 3])
def test_radiometric_scan_empty_valid(d):
    rs = RadiometricScan.empty(B=6, K=5, J=10, d=d, pose=Pose.identity(d))
    assert (rs.B, rs.K, rs.J, rs.dim) == (6, 5, 10, d)
    assert not rs.hit.any() and np.isnan(rs.r_path).all() and (rs.power == 0).all()
    assert (rs.sub_idx[:, :5] == np.arange(5)).all()


# ------------------------------------------
# test_radiometric_scan_rejects_bad_weights
# ------------------------------------------
# Sub-ray weights that do not sum to one are rejected.
def test_radiometric_scan_rejects_bad_weights():
    rs = RadiometricScan.empty(B=2, K=3, J=3, d=2, pose=Pose.identity(2))
    with pytest.raises(ValueError, match="w"):
        RadiometricScan(**{**rs.fields(), "w": np.full((2, 3), 0.5)})


# ------------------------------------------
# test_scan_nan_iff_not_detected
# ------------------------------------------
# A scan's range is NaN exactly when nothing was detected.
def test_scan_nan_iff_not_detected():
    rs = RadiometricScan.empty(B=3, K=1, J=1, d=2, pose=Pose.identity(2))
    ok = Scan(detected=np.array([True, False, True]), range=np.array([1.0, np.nan, 2.0]),
              amplitude=np.ones((3, 2)), range_raw=np.array([1.0, 7.0, 2.0]),
              fire_time=np.zeros(3), truth=rs)
    assert ok.B == 3
    with pytest.raises(ValueError, match="range"):
        Scan(detected=np.array([True, True, True]), range=np.array([1.0, np.nan, 2.0]),
             amplitude=np.ones((3, 2)), range_raw=np.zeros(3), fire_time=np.zeros(3), truth=rs)


# ------------------------------------------
# test_beam_prediction_shapes
# ------------------------------------------
# Beam prediction shapes.
def test_beam_prediction_shapes():
    B = 4
    bp = BeamPrediction(A=np.ones((B, 2)), sigma_n=np.ones((B, 2)), snr=np.ones((B, 2)),
                        P_D=np.full(B, 0.9), P_FA=np.full(B, 1e-4), r_bar=np.ones(B),
                        sigma_R=np.full(B, 0.01), out_prob=np.zeros((B, 6)),
                        out_shift=np.zeros((B, 6)), hit_any=np.ones(B, bool),
                        mixed=np.zeros(B, bool), smooth=np.ones(B, bool))
    assert bp.B == 4 and bp.n_out == 6
    with pytest.raises(ValueError, match="P_D"):
        BeamPrediction(**{**bp.fields(), "P_D": np.full(B, 1.5)})


# ------------------------------------------
# test_mle_result
# ------------------------------------------
# The pose-estimate result record.
def test_mle_result():
    r = MLEResult(xi=np.zeros(3), success=True, n_iter=5, loglik=-1.0, message="ok")
    assert r.success


# ------------------------------------------
# test_return_axis_must_be_K_or_2K
# ------------------------------------------
# The return axis holds K or 2K slots.
def test_return_axis_must_be_K_or_2K():
    with pytest.raises(ValueError, match="J"):
        RadiometricScan.empty(B=2, K=3, J=4, d=2)
    assert RadiometricScan.empty(B=2, K=3, J=6, d=2).J == 6
    assert RadiometricScan.empty(B=2, K=3).J == 3          # J defaults to K, pose to identity


# ------------------------------------------
# test_dimension_must_be_2_or_3
# ------------------------------------------
# Records accept only 2D and 3D.
def test_dimension_must_be_2_or_3():
    with pytest.raises(ValueError, match="d"):
        RadiometricScan.empty(B=2, K=1, d=4)
    with pytest.raises(ValueError, match="d"):
        _hits(d=4)


# ------------------------------------------
# test_unused_return_conventions_enforced
# ------------------------------------------
# Unused return slots must be empty.
def test_unused_return_conventions_enforced():
    rs = RadiometricScan.empty(B=2, K=2, d=2)
    bad_r = rs.r_path.copy()
    bad_r[0, 0] = 1.0                                       # miss with finite r_path
    with pytest.raises(ValueError, match="r_path"):
        RadiometricScan(**{**rs.fields(), "r_path": bad_r})
    bad_id = rs.object_id.copy()
    bad_id[0, 0] = 0                                        # miss with a real id
    with pytest.raises(ValueError, match="object_id"):
        RadiometricScan(**{**rs.fields(), "object_id": bad_id})


# ------------------------------------------
# test_hit_entries_validated
# ------------------------------------------
# Hit entries are validated.
def test_hit_entries_validated():
    rs = RadiometricScan.empty(B=1, K=1, d=2)
    f = rs.fields()
    f.update(hit=np.array([[True]]), r_path=np.array([[2.0]]), power=np.array([[1e-7]]),
             normal=np.array([[[-1.0, 0.0]]]), cos_inc=np.array([[1.0]]),
             object_id=np.array([[0]]), material_id=np.array([[0]]), surface_id=np.array([[0]]))
    RadiometricScan(**f)
    with pytest.raises(ValueError, match="normal"):
        RadiometricScan(**{**f, "normal": np.array([[[-2.0, 0.0]]])})
    with pytest.raises(ValueError, match="cos_inc"):
        RadiometricScan(**{**f, "cos_inc": np.array([[1.5]])})
    with pytest.raises(ValueError, match="dir_world"):
        RadiometricScan(**{**f, "dir_world": np.array([[[2.0, 0.0]]])})


# ------------------------------------------
# test_hitbatch_miss_ids_and_unit_normals
# ------------------------------------------
# Misses carry id -1 and hits carry unit normals.
def test_hitbatch_miss_ids_and_unit_normals():
    h = _hits(n=2)
    with pytest.raises(ValueError, match="surface_id"):
        HitBatch(**{**h.fields(), "hit": np.array([True, False])})   # miss must carry ids = -1
    with pytest.raises(ValueError, match="normal"):
        HitBatch(**{**h.fields(), "normal": np.full((2, 2), 3.0)})


# ------------------------------------------
# test_scan_rejects_infinite_and_mismatched_times
# ------------------------------------------
# Infinite or mismatched fire times are rejected.
def test_scan_rejects_infinite_and_mismatched_times():
    rs = RadiometricScan.empty(B=2, K=1)
    kw = {"detected": np.array([True, True]), "amplitude": np.ones((2, 2)), "range_raw": np.zeros(2),
              "fire_time": np.zeros(2), "truth": rs}
    with pytest.raises(ValueError, match="finite"):
        Scan(range=np.array([1.0, np.inf]), **kw)
    with pytest.raises(ValueError, match="fire_time"):
        Scan(range=np.array([1.0, 2.0]), **{**kw, "fire_time": np.ones(2)})


# ------------------------------------------
# test_specular_slot_semantics
# ------------------------------------------
# What the specular slots may contain.
def test_specular_slot_semantics():
    rs = RadiometricScan.empty(B=1, K=1, J=2)
    f = rs.fields()
    hit = np.array([[False, True]])
    f.update(hit=hit, r_path=np.array([[np.nan, 4.0]]), power=np.array([[0.0, 1e-8]]),
             normal=np.array([[[0.0, 0.0], [0.0, -1.0]]]), cos_inc=np.array([[0.0, 1.0]]),
             object_id=np.array([[-1, 0]]), material_id=np.array([[-1, 0]]), surface_id=np.array([[-1, 2]]))
    RadiometricScan(**{**f, "bounce": np.array([[False, True]])})
    with pytest.raises(ValueError, match="bounce"):
        RadiometricScan(**{**f, "bounce": np.array([[False, False]])})      # hit in bounced slot
    with pytest.raises(ValueError, match="bounce"):
        RadiometricScan(**{**rs.fields(), "bounce": np.array([[True, False]])})   # direct slot


# ------------------------------------------
# test_out_prob_must_have_six_columns
# ------------------------------------------
# Wrap probabilities have six columns.
def test_out_prob_must_have_six_columns():
    B = 2
    base = {"A": np.ones((B, 2)), "sigma_n": np.ones((B, 2)), "snr": np.ones((B, 2)),
                "P_D": np.full(B, 0.5), "P_FA": np.full(B, 1e-4), "r_bar": np.ones(B), "sigma_R": np.ones(B),
                "hit_any": np.ones(B, bool), "mixed": np.zeros(B, bool), "smooth": np.ones(B, bool)}
    with pytest.raises(ValueError, match="out_prob"):
        BeamPrediction(out_prob=np.zeros((B, 4)), out_shift=np.zeros((B, 4)), **base)
