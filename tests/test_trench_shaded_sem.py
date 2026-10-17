import cv2
import numpy as np
import pytest

from angle_cal.trench_analysis import TrenchOptions
from angle_cal.trench_guides import points_in_guides
from angle_cal.trench_workflow import recognize_from_guides


def shaded_sem():
    yy, xx = np.mgrid[:360, :440]
    image = np.full(xx.shape, 170., dtype=float)
    image[yy < 40] = 55
    void = (xx > 150) & (xx < 250) & (yy >= 40) & (yy < 270)
    # SEM illumination gradient: the inner left wall is brighter than much of
    # the outer material, even though the center is a continuous darker cavity.
    shade = 78+90*np.exp(-(xx-150)/25)+30*(270-yy)/230
    image[void] = shade[void]
    for x in (150, 250):
        image += 55*np.exp(-((xx-x)/3)**2)*((yy >= 42) & (yy < 268))
    image[330:] = 0  # acquisition footer, outside the requested contour/context
    image = cv2.GaussianBlur(image.astype(np.float32), (5, 5), .8)
    return np.clip(image, 0, 255).astype(np.uint8)


def guides():
    return (dict(points=[[100, 40], [150, 40], [150, 100], [150, 240], [150, 270],
                         [200, 270], [250, 270], [250, 240], [250, 100], [250, 40], [300, 40]],
                 half_width_px=10),)


@pytest.mark.parametrize('variant', ['normal', 'annotation', '16bit', 'inverted'])
def test_shaded_sem_keeps_bright_wall_instead_of_dark_threshold_crossing(variant):
    image = shaded_sem()
    if variant == 'annotation':
        cv2.putText(image, '1.0', (174, 210), cv2.FONT_HERSHEY_SIMPLEX, .4, 255, 1)
    if variant == '16bit':image = image.astype(np.uint16)*200
    if variant == 'inverted':image = 255-image
    r = recognize_from_guides(image, TrenchOptions((0, 0, 440, 360), wall_guides=guides()))
    middle = (r.y > 80) & (r.y < 240)
    assert np.median(r.left[middle]) == pytest.approx(150, abs=4)
    assert np.median(r.right[middle]) == pytest.approx(250, abs=4)
    assert r.depth_px == pytest.approx(230, abs=5)
    assert points_in_guides(np.column_stack((r.left, r.y)), guides()).all()
    assert points_in_guides(np.column_stack((r.right, r.y)), guides()).all()


def test_remote_footer_cannot_change_guided_measurements():
    image = shaded_sem()
    changed = image.copy();changed[330:] = 255
    results = [recognize_from_guides(im, TrenchOptions((0, 0, 440, 360), wall_guides=guides()))
               for im in (image, changed)]
    for key in ('y', 'left', 'right', 'sample_left', 'sample_right'):
        np.testing.assert_array_equal(getattr(results[0], key), getattr(results[1], key))
    assert results[0].depth_px == results[1].depth_px


def test_two_different_trenches_are_not_one_shaded_cavity():
    image = np.full((320, 440), 170, np.uint8)
    image[:40] = 40
    image[40:270, 70:150] = 50
    image[40:270, 270:350] = 50
    g = dict(points=[[70, 50], [70, 250], [350, 250], [350, 50]], half_width_px=8)
    with pytest.raises(ValueError, match='곡선'):
        recognize_from_guides(image, TrenchOptions((0, 0, 440, 320), wall_guides=(g,)))
