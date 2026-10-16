import json
import numpy as np
import pytest

from angle_cal.trench_analysis import TrenchOptions,analyze_trench,export_trench_csv,trench_gfe_tsv
from angle_cal.trench_contour import trace_guide
from test_trench_corner import rounded_trench
from test_trench_analysis import trench_image


def guide(points,radius=18,**extra):
    return dict(points=points,half_width_px=radius,**extra)


def rim_image(noise=0):
    xx=np.arange(240)
    row=np.where((xx>80)&(xx<160),25.,130.)
    row+=120*np.exp(-((xx-79)/2.2)**2)+120*np.exp(-((xx-161)/2.2)**2)
    row+=125*np.exp(-((xx-65)/1.2)**2)  # unrelated brighter material line
    image=np.tile(row,(240,1));image[:20]=25;image[220:]=130
    image+=np.random.default_rng(19).normal(0,noise,image.shape)
    return np.clip(image,0,255).astype(np.uint8)


def whole_u():
    return guide([[79,30],[79,190],[79,215],[100,220],[140,220],[161,215],[161,190],[161,30]],22)


def test_legacy_side_labels_do_not_change_any_measured_coordinate():
    image=rim_image()
    guides=(guide([[79,30],[79,200]],22,side='left'),guide([[161,30],[161,200]],22,side='right'))
    a=analyze_trench(image,TrenchOptions((40,20,160,215),wall_guides=guides))
    swapped=(dict(guides[0],side='right'),dict(guides[1],side='left'))
    b=analyze_trench(image,TrenchOptions((40,20,160,215),wall_guides=swapped))
    for key in ('y','left','right','sample_left','sample_right'):
        assert np.array_equal(getattr(a,key),getattr(b,key))
    assert a.depth_px==b.depth_px


def test_rough_guide_does_not_choose_a_brighter_material_line():
    x=np.arange(240)
    row=np.where((x>80)&(x<160),25.,130.)
    row+=45*np.exp(-((x-79)/1.8)**2)+90*np.exp(-((x-161)/2.)**2)
    row+=125*np.exp(-((x-65)/1.2)**2)
    image=np.tile(row,(240,1));image[:20]=25;image[220:]=130
    image=np.clip(image,0,255).astype(np.uint8)
    guides=(guide([[60,30],[60,200]],30),guide([[161,30],[161,200]],15))
    result=analyze_trench(image,TrenchOptions((40,20,160,215),wall_guides=guides))
    middle=(result.y>50)&(result.y<190)
    assert np.median(result.left[middle])==pytest.approx(79,abs=1)


@pytest.mark.parametrize('variant',['normal','reversed','bright','16bit','noise'])
def test_one_ordered_u_curve_finds_both_walls_and_floor(variant):
    image=rim_image(2 if variant=='noise' else 0)
    g=whole_u()
    if variant=='reversed':g=dict(g,points=g['points'][::-1])
    if variant=='bright':image=255-image
    if variant=='16bit':image=image.astype(np.uint16)*201
    result=analyze_trench(image,TrenchOptions((40,20,160,215),bright_trench=variant=='bright',wall_guides=(g,)))
    middle=(result.y>50)&(result.y<190)
    assert np.median(result.left[middle])==pytest.approx(79,abs=1)
    assert np.median(result.right[middle])==pytest.approx(161,abs=1)
    assert result.depth_px==pytest.approx(199,abs=2)
    assert len(result.guide_traces)==1 and result.guide_traces[0]['trace_coverage']>.95


@pytest.mark.parametrize('variant',['normal','rim','layers','blur','noise'])
def test_field_corner_wall_floor_are_traced_as_one_contour(variant):
    image=rounded_trench(15,30,rim=variant=='rim',layers=variant=='layers',
                         blur=.6 if variant=='blur' else 0,noise=2 if variant=='noise' else 0)
    g=guide([[90,45],[132,45],[148,50],[160,70],[160,315],[160,345],[190,350],
             [230,350],[260,345],[260,315],[260,80],[268,54],[287,45],[330,45]],18)
    result=analyze_trench(image,TrenchOptions((60,45,300,335),wall_guides=(g,)))
    middle=(result.y>110)&(result.y<300)
    # The resolved bright crest is outside the nominal dark-space boundary.
    expected_cd=103.5 if variant=='rim' else 100
    assert np.median(result.right[middle]-result.left[middle])==pytest.approx(expected_cd,abs=2)
    assert result.depth_px==pytest.approx(304,abs=2)
    assert result.guide_traces[0]['trace_coverage']>.8
    assert all(item['status']=='valid' for item in result.field_angles)


def test_unrelated_flat_band_never_becomes_a_measured_contour():
    g=guide([[20,30],[20,200]],2)
    with pytest.raises(ValueError,match='가이드'):
        analyze_trench(rim_image(),TrenchOptions((10,20,220,215),wall_guides=(g,)))


def test_narrow_corridor_on_material_line_cannot_borrow_cavity_across_another_rim():
    g=guide([[65,30],[65,200]],2)
    with pytest.raises(ValueError,match='가이드'):
        analyze_trench(rim_image(),TrenchOptions((40,20,160,215),wall_guides=(g,)))


@pytest.mark.parametrize('one_curve',[False,True])
def test_bright_inclusion_does_not_split_the_outer_trench_contour(one_curve):
    image=trench_image()
    image[90:135,96:105]=205
    if one_curve:
        guides=(guide([[69,30],[69,205],[74,220],[124,220],[130,205],[130,30]],12),)
    else:
        guides=(guide([[69,30],[69,210]],12),guide([[130,30],[130,210]],12))
    result=analyze_trench(image,TrenchOptions((30,20,140,225),wall_guides=guides))
    middle=(result.y>70)&(result.y<160)
    assert np.median(result.right[middle]-result.left[middle])==pytest.approx(60,abs=1)
    assert result.depth_px==pytest.approx(201,abs=2)
    assert not np.any(np.diff(result.y)>1.5)


def test_missing_evidence_is_reported_without_guide_coordinate_substitution():
    image=trench_image().astype(np.float32)
    image[112:114]=205  # erased image evidence, not a known wall
    g=guide([[69,30],[69,205]],8)
    trace=trace_guide(image,g,80)
    assert trace['trace_coverage']<1
    assert trace['detected_points']<trace['requested_points']
    result=analyze_trench(image,TrenchOptions((30,20,140,225),step_px=.5,wall_guides=(g,)))
    gaps=np.flatnonzero(np.diff(result.y)>1.5)
    assert len(gaps)
    for i in gaps:
        assert not np.any((result.sample_y>result.y[i])&(result.sample_y<result.y[i+1]))
    with pytest.raises(ValueError,match='미검출'):
        trench_gfe_tsv(result,.5)


def test_trace_evidence_and_method_are_saved_in_json_and_csv(tmp_path):
    result=analyze_trench(rim_image(),TrenchOptions((40,20,160,215),wall_guides=(whole_u(),)))
    data=json.loads(json.dumps(result.to_dict(.5),allow_nan=False))
    assert data['guide_method']=='ordered_normal_contour_then_cavity_classification'
    assert len(data['guide_traces'][0]['segments'])>=1
    csv=tmp_path/'contour.csv';export_trench_csv(csv,result,.5)
    assert 'contour_trace' in csv.read_text(encoding='utf-8-sig')
