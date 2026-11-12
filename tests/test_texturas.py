import numpy as np
from types import SimpleNamespace
from closet.texturas import analyze_texture
from closet.atributos import characterize


def rgb(values):return np.repeat(np.asarray(values,dtype=np.uint8)[...,None],3,axis=2)


def test_smooth_fabric_and_gradual_illumination_are_not_microtexture():
    y,x=np.indices((160,160));a=rgb(100+60*x/159+10*np.sin(y/45))
    d=analyze_texture(a,np.ones((160,160),bool))
    assert d['label']=='Sin microtextura visible'
    assert d['pattern_label']=='Sin patrón repetitivo visible'


def test_small_scale_gray_texture_not_overridden_by_legacy_plain_label():
    y,x=np.indices((180,180));a=rgb(190+25*((x//2+y//2)%2))
    m=np.ones((180,180),bool)
    norm=SimpleNamespace(rgb=a,mascara=m,nucleo=m)
    d=characterize(norm,{'estampado':'liso'},source_rgb=a,source_mask=m)
    assert d['texture']!='Sin microtextura visible'
    assert d['texture_analysis']['reliable']
    assert d['pattern']['label']!='Liso'
    assert d['texture_analysis']['metrics']['detail_density']>.1


def test_background_checkerboard_cannot_create_texture_in_plain_shirt():
    y,x=np.indices((180,180));a=rgb(40+180*((x//3+y//3)%2));m=np.zeros((180,180),bool)
    m[35:145,45:135]=True;a[m]=140
    d=analyze_texture(a,m)
    assert d['label']=='Sin microtextura visible'


def test_stripes_and_checks_are_separate_visible_patterns():
    y,x=np.indices((180,180));m=np.ones((180,180),bool)
    vertical=analyze_texture(rgb(50+150*((x//8)%2)),m)
    horizontal=analyze_texture(rgb(50+150*((y//8)%2)),m)
    checks=analyze_texture(rgb(50+150*((x//8+y//8)%2)),m)
    assert vertical['pattern_label']=='Rayas verticales'
    assert horizontal['pattern_label']=='Rayas horizontales'
    assert checks['pattern_label']=='Cuadros o rejilla'


def test_insufficient_resolution_is_not_called_plain():
    a=np.full((18,18,3),150,np.uint8)
    d=analyze_texture(a,np.ones((18,18),bool))
    assert d['label']=='No concluyente' and not d['reliable']


def test_uses_native_input_instead_of_blurred_normalization():
    y,x=np.indices((160,160));source=rgb(200+30*((x//2+y//2)%2));m=np.ones((160,160),bool)
    smooth=rgb(np.full((160,160),210))
    norm=SimpleNamespace(rgb=smooth,mascara=m,nucleo=m)
    d=characterize(norm,{'estampado':'liso'},source_rgb=source,source_mask=m)
    assert d['texture']!='Sin microtextura visible'
