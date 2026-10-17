"""Image-supported continuity must reduce grains without flattening geometry."""
import cv2
import numpy as np
import pytest

from angle_cal.trench_contour import trace_guide, contour_row_targets
from angle_cal.trench_guides import points_in_guides
from test_trench_corner import rounded_trench


def sem_wall(kind='straight', noisy=False):
    yy, xx = np.mgrid[:320, :260]
    shape = np.zeros(320)
    if kind == 'bow':shape = -16*np.exp(-((np.arange(320)-160)/55)**2)
    if kind == 'notch':shape[145:166] = -8
    if kind == 'taper':shape = .04*(np.arange(320)-160)
    left = 90+shape[:, None]
    image = np.full(xx.shape, 155., float)
    inside = (xx > left) & (xx < 190)
    shade = 42+108*np.exp(-np.maximum(xx-left, 0)/19)
    image[inside] = shade[inside]
    image += 62*np.exp(-((xx-left)/2.5)**2)
    image = cv2.GaussianBlur(image.astype(np.float32), (3, 3), .6)
    if noisy:
        rng = np.random.default_rng(713)
        image += rng.normal(0, 9, image.shape)
        for row in (70, 115, 205, 245):image[row, 80:110] = 245
    return np.clip(image, 0, 255).astype(np.float32), left[:, 0]


def trace(image, shift=0, reverse=False):
    points = [[90+shift, 30], [90+shift, 290]]
    g = dict(points=points[::-1] if reverse else points, half_width_px=30)
    t = trace_guide(image, g, 75, 'max_gradient', local_contrast=True, material_level=155)
    assert points_in_guides(t['points'], (g,)).all()
    rows = contour_row_targets([t])
    y = np.array(sorted(rows))
    x = np.array([np.median(rows[k]) for k in y])
    return t, y, x


@pytest.mark.parametrize('kind', ['straight', 'bow', 'notch', 'taper'])
def test_consensus_rejects_grains_but_retains_true_shape(kind):
    clean, expected = sem_wall(kind)
    noisy, _ = sem_wall(kind, True)
    _, cy, cx = trace(clean)
    _, y, x = trace(noisy)
    assert len(y) >= 240
    error = x-np.interp(y, cy, cx)
    assert np.sqrt(np.mean(error**2)) < 1.1
    assert np.percentile(np.abs(error), 95) < 2
    assert np.median(np.abs(x-expected[y.astype(int)])) < 3
    assert np.median(np.abs(cx-expected[cy.astype(int)])) < 3
    if kind in ('bow', 'notch'):
        baseline = np.median(x[(y > 50) & (y < 90)])
        center = np.median(x[(y > 150) & (y < 161)])
        amplitude = baseline-center
        assert amplitude >= (13 if kind == 'bow' else 6)
        assert amplitude <= (19 if kind == 'bow' else 10)
    if kind == 'taper':
        assert np.polyfit(y, x, 1)[0] == pytest.approx(.04, abs=.007)


def test_guide_location_and_drawing_direction_do_not_set_measured_wall():
    image, _ = sem_wall('bow', True)
    _, y, x = trace(image)
    for shift, reverse in ((5, False), (-5, False), (0, True)):
        _, other_y, other_x = trace(image, shift, reverse)
        assert np.percentile(np.abs(x-np.interp(y, other_y, other_x)), 95) < 1


def test_blank_band_is_not_reconstructed_by_neighbor_consensus():
    image, _ = sem_wall('straight', True)
    image[140:160] = 155
    t, y, x = trace(image)
    assert not np.any((y >= 142) & (y <= 157))
    assert len(t['segments']) >= 2
    assert t['detected_points'] < t['requested_points']


def test_flat_noisy_region_cannot_be_filled_with_a_straight_guide():
    image = np.full((320, 260), 155, np.float32)
    image += np.random.default_rng(13).normal(0, 2, image.shape)
    t, y, x = trace(image)
    assert len(t['points']) == 0


@pytest.mark.parametrize('radius', [8, 15, 30, 60])
def test_small_and_large_entrance_arcs_are_not_flattened(radius):
    center = np.array([160-radius, 45+radius])
    angle = np.linspace(-np.pi/2, 0, 7)
    arc = center+radius*np.column_stack((np.cos(angle), np.sin(angle)))
    points = np.vstack(([60, 45], arc, [160, 300]))
    g = dict(points=points.tolist(), half_width_px=8)
    t = trace_guide(rounded_trench(radius, noise=2, blur=.6).astype(np.float32),
                    g, 80, 'max_gradient', local_contrast=True, material_level=205)
    p = np.asarray(t['points'])
    circular = p[(p[:, 0] > center[0]+2) & (p[:, 1] < center[1]-2)]
    assert len(circular) >= 5
    error = np.abs(np.linalg.norm(circular-center, axis=1)-radius)
    assert np.percentile(error, 95) < max(1, radius*.1)
    assert points_in_guides(p, (g,)).all()
    assert t['preprocessing']['missing_policy'] == 'bounded_short_gap_connection'


def test_short_missing_strip_is_connected_and_labeled_as_interpolation():
    image, _ = sem_wall('straight', True)
    image[145:148] = 155
    t, y, x = trace(image)
    assert np.all(np.isin(np.arange(142,151),y))
    assert t['interpolated_points'] > 0
    assert len(t['point_sources']) == len(t['points']) == len(t['edge_spreads_px'])
    assert t['detected_points']+t['interpolated_points'] == len(t['points'])
    for source, spread in zip(t['point_sources'],t['edge_spreads_px']):
        if source == 'interpolated':assert spread is None
    assert np.ptp(x[(y>=142)&(y<=151)]) < 1


def test_short_joins_do_not_extrapolate_or_jump_to_another_wall():
    from angle_cal.trench_contour import _connect_short_gaps
    distance=np.arange(30,dtype=float)
    g=dict(points=[[30,0],[30,29]],half_width_px=15)
    measured={i:[30.,float(i)] for i in range(3,27) if i not in (12,13,14)}
    joined, joins=_connect_short_gaps(measured,distance,g,(50,60))
    assert len(joins)==1 and set(joined)==set(range(3,27))
    # Two vertical branches cannot be joined across a transverse jump.
    for i in range(15,27):measured[i][0]=43.
    joined, joins=_connect_short_gaps(measured,distance,g,(50,60))
    assert not joins and joined==measured


def test_short_join_stays_inside_nonconvex_guide_corridor():
    from angle_cal.trench_contour import _connect_short_gaps
    distance=np.arange(30,dtype=float)
    measured={i:[30.,float(i)] for i in range(3,27) if i not in range(11,19)}
    g=dict(points=[[30,0],[30,10],[40,15],[30,20],[30,29]],half_width_px=2)
    joined, joins=_connect_short_gaps(measured,distance,g,(50,60))
    assert not joins and joined==measured


def test_shallow_single_step_on_vertical_wall_does_not_drop_cd_row():
    points=[[30.,float(y)] for y in range(10,22)]
    points[4]=[30.,14.7];points[5]=[32.,15.15]
    rows=contour_row_targets([dict(segments=[points])])
    assert 15 in rows and 30 < rows[15][0] < 32
    # A genuinely horizontal/slightly tilted Field is not a wall target.
    field=[[float(x),10+x*.1] for x in range(50)]
    field_rows=contour_row_targets([dict(segments=[field])])
    assert 12 not in field_rows
