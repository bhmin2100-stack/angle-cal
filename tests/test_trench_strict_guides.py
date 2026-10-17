import numpy as np
import pytest

from angle_cal.trench_analysis import TrenchOptions
from angle_cal.trench_guides import points_in_guides, guide_intervals, in_bands
from angle_cal.trench_workflow import recognize_from_guides
from test_trench_analysis import trench_image


def guide(points, radius=4):
    return dict(points=points, half_width_px=radius)


def test_one_wall_cannot_borrow_opposite_wall_from_outside_corridor():
    with pytest.raises(ValueError, match='곡선'):
        recognize_from_guides(trench_image(), TrenchOptions((0, 0, 200, 260),
                             wall_guides=(guide([[69, 40], [69, 180]]),)))


@pytest.mark.parametrize('radius', [1.5, 4, 10])
def test_wall_profile_and_cd_samples_stay_in_swept_scope_after_smoothing(radius):
    guides = (guide([[69, 40], [69, 180]], radius), guide([[130, 40], [130, 180]], radius))
    r = recognize_from_guides(trench_image(), TrenchOptions((0, 0, 200, 260), step_px=.6, smooth_px=19, wall_guides=guides))
    for x, y in ((r.left, r.y), (r.right, r.y), (r.sample_left, r.sample_y), (r.sample_right, r.sample_y)):
        assert points_in_guides(np.column_stack((x, y)), guides).all()
    assert r.y.min() >= 40-radius and r.y.max() <= 180+radius
    assert r.depth_px is None
    assert all(item['status'] == 'unconfirmed' and not item['points'] for item in r.field_angles)


def test_depth_is_not_inferred_beyond_guide_endcaps():
    guides = (guide([[69, 30], [69, 217]], 2), guide([[130, 30], [130, 217]], 2))
    r = recognize_from_guides(trench_image(), TrenchOptions((0, 0, 200, 260), wall_guides=guides))
    assert r.y[-1] <= 219 and r.depth_px is None


def test_bulk_scope_membership_matches_displayed_disk_sweep():
    g = guide([[20, 30], [70, 90], [25, 150], [100, 190]], 8)
    points = np.random.default_rng(13).uniform([0, 0], [140, 220], (1500, 2))
    expected = [in_bands(x, guide_intervals(g, y)) for x, y in points]
    np.testing.assert_array_equal(points_in_guides(points, (g,)), expected)
