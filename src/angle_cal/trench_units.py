"""Display-only metric lengths; simulation coordinates always remain Å."""
import math


def format_metric_length(value_px, nm_per_px, unit='auto'):
    if value_px is None:return '미확정'
    if unit not in ('auto','um','nm','angstrom'):raise ValueError('지원하지 않는 길이 단위입니다.')
    if not math.isfinite(nm_per_px) or nm_per_px<=0:return f'{value_px:.5g} px'
    nm=value_px*nm_per_px
    if unit=='auto':unit='um' if abs(nm)>=1000 else 'nm' if abs(nm)>=1 or nm==0 else 'angstrom'
    factor,label={'um':(.001,'µm'),'nm':(1,'nm'),'angstrom':(10,'Å')}[unit]
    return f'{nm*factor:.5g} {label}'
