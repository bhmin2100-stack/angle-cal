from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from angle_cal.trench_panel import TrenchGraph
from angle_cal.trench_plot import shape_series


def result_with_floor():
    y = np.arange(10., 91.)
    return SimpleNamespace(
        options=SimpleNamespace(roi=(0, 10, 100, 110)),
        y=y, left=np.full_like(y, 40), right=np.full_like(y, 60),
        depth_px=90., guide_traces=[dict(segments=[
            [[20, 5], [40, 10], [40, 90], [44, 97], [50, 100],
             [56, 97], [60, 90], [60, 10], [80, 2]]])],
        field_angles=[dict(status='valid', angle_status='unconfirmed', points=[[20, 5], [40, 10]]),
                      dict(status='valid', angle_status='unconfirmed', points=[[60, 10], [80, 2]])],
        local_bowing=[])


def test_full_measured_contour_keeps_both_fields_and_curved_bottom():
    r = result_with_floor()
    series = shape_series(r)
    np.testing.assert_array_equal(np.column_stack(series[0][:2]), r.guide_traces[0]['segments'][0])
    fields = [s for s in series if s[2] == '#268b48']
    assert len(fields) == 2  # Field geometry does not require a valid angle.
    assert max(np.max(s[1]) for s in series) == 100 > r.y[-1]
    assert not any(s[3] for s in series)  # No flat substitute over the curved floor.
    assert series[-2][0] is r.left and series[-1][0] is r.right
    np.testing.assert_array_equal(series[-1][0] - series[-2][0], r.right - r.left)


def test_missing_trace_segments_stay_separate():
    r = result_with_floor()
    r.guide_traces[0]['segments'] = [r.guide_traces[0]['segments'][0][:4],
                                     r.guide_traces[0]['segments'][0][5:]]
    traces = [s for s in shape_series(r) if s[2] == '#8f45a4' and not s[3]]
    assert len(traces) == 2
    assert traces[0][0][-1] == 44 and traces[1][0][0] == 56


def test_known_depth_without_floor_trace_uses_dashed_reference():
    r = result_with_floor()
    r.guide_traces = []
    series = shape_series(r)
    reference = [s for s in series if s[3]]
    assert len(reference) == 1
    assert max(reference[0][1]) == r.options.roi[1] + r.depth_px


def test_truncated_trench_does_not_get_invented_bottom_or_field():
    r = result_with_floor()
    r.guide_traces = []
    r.depth_px = None
    r.field_angles[0]['status'] = 'unconfirmed'
    series = shape_series(r)
    assert not any(s[3] for s in series)
    assert sum(s[2] == '#268b48' for s in series) == 1
    assert max(np.max(s[1]) for s in series) == r.y[-1]


@pytest.mark.parametrize('factor,unit', [(1., 'px'), (.5, 'nm'), (5., 'Å')])
def test_graph_bounds_include_field_and_floor_with_same_pixel_geometry(factor, unit):
    app = QApplication.instance() or QApplication([])
    graph = TrenchGraph()
    graph.resize(420, 240)
    graph.result = result_with_floor()
    graph.factor, graph.unit = factor, unit
    graph.show()
    app.processEvents()
    graph.grab()
    assert graph.plot_y_bounds == (2., 100.)
    assert not graph.show_entrance  # No angle or radius controls/annotations return.
    graph.close()
