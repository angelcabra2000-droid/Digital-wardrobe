"""Descriptores geométricos y de patrón; estimaciones visuales, no materiales textiles."""
import numpy as np
from .segmentacion import filtro_caja
from .texturas import analyze_texture

COLORS = {
    'negro':'#141414', 'blanco':'#f5f5f5', 'gris':'#808080', 'beige':'#d7c3a0',
    'marrón':'#795338', 'rojo':'#c81e28', 'naranja':'#f0821e', 'amarillo':'#efcf32',
    'verde':'#329646', 'turquesa':'#28a0a0', 'azul':'#2850c8', 'morado':'#6e3296',
    'rosa':'#ec83ad',
}
COLOR_FAMILIES = {
    'gris oscuro':'gris', 'gris claro':'gris', 'crema':'beige', 'caqui':'beige',
    'vino':'rojo', 'mostaza':'amarillo', 'verde oliva':'verde', 'verde oscuro':'verde',
    'celeste':'azul', 'azul marino':'azul', 'azul jean':'azul',
    'jean claro':'azul', 'fucsia':'rosa',
}
UNKNOWN = 'Sin determinar'
PATTERNS = ('Liso', 'Rayas verticales', 'Rayas horizontales', 'Cuadros', 'Trama repetitiva',
            'Otro estampado', UNKNOWN)
TEXTURES = ('Lisa', 'Fina', 'Marcada', 'Repetitiva', 'Irregular', UNKNOWN)
PATTERN_GROUPS = {'Cuadros o rejilla':'Cuadros', 'Rayas':'Trama repetitiva',
                  'Trama fina repetitiva':'Trama repetitiva'}
OTHER_PRINTS = {'Flores', 'Hojas', 'Palmeras', 'Animal print', 'Lunares', 'Gráfico o texto', 'Geométrico'}
TEXTURE_GROUPS = {
    'Sin microtextura visible':'Lisa', 'Textura fina visible':'Fina',
    'Textura marcada / detalle contrastado':'Marcada', 'Microtextura repetitiva':'Repetitiva',
    'Textura visual irregular':'Irregular',
}


def color_family(features, color):
    family = COLOR_FAMILIES.get(color, color)
    if family == 'gris' and all(key in features for key in ('color_L', 'color_a', 'color_b_lab')):
        light, a, b = (features[key] for key in ('color_L', 'color_a', 'color_b_lab'))
        if a >= 3 and b >= 3 and np.hypot(a, b) >= 9 and light < 85:
            return 'beige' if light >= 49 else 'marrón'
    return family if family in COLORS else UNKNOWN


def catalog_attributes(attributes, features):
    """Proyecta análisis actuales y anteriores sobre las categorías del clóset."""
    result = dict(attributes)
    manual = result.get('color_override')
    result['color_family'] = manual if manual in COLORS else color_family(
        features, result.get('color') or features.get('color_nombre', result.get('color_family', '')))
    print_info = result.get('print', {})
    label = (print_info.get('label') if print_info.get('method') == 'CLIP' else None
             ) or result.get('pattern_group') or result.get('pattern', {}).get('label') or print_info.get('label')
    if label in OTHER_PRINTS:
        label = 'Otro estampado'
    if label == 'Sin estampado' or (label == 'Sin patrón repetitivo visible' and result.get('texture') == 'Sin microtextura visible'):
        label = 'Liso'
    label = PATTERN_GROUPS.get(label, label)
    result['pattern_category'] = label if label in PATTERNS else UNKNOWN
    texture = result.get('texture')
    result['texture_category'] = TEXTURE_GROUPS.get(texture, UNKNOWN)
    return result


def periodicity(signal, mask, axis):
    values=[]
    for shift in range(3,49):
        if axis==1:
            a,b=signal[:,:-shift],signal[:,shift:]
            valid=mask[:,:-shift]&mask[:,shift:]
        else:
            a,b=signal[:-shift],signal[shift:]
            valid=mask[:-shift]&mask[shift:]
        if valid.sum()<80:
            values.append(0.0);continue
        aa,bb=a[valid].ravel(),b[valid].ravel()
        denom=float(np.sqrt((aa*aa).sum()*(bb*bb).sum()))
        values.append(float((aa*bb).sum()/denom) if denom>1e-10 else 0)
    peaks=[values[i] for i in range(1,len(values)-1) if values[i]>values[i-1] and values[i]>values[i+1]]
    return bool(peaks and max(peaks)>.48 and min(values)<-.12)


def pattern(norm, features):
    m=norm.nucleo if norm.nucleo.any() else norm.mascara
    rgb=norm.rgb.astype(float)/255
    denominator=filtro_caja(m.astype(float),8)
    smooth=np.stack([filtro_caja(rgb[...,channel]*m,8) for channel in range(3)],axis=-1)/np.maximum(denominator[...,None],1e-6)
    high=(rgb-smooth)*m[...,None]
    high[denominator<.8]=0
    inner=m&(denominator>.8)
    rms=float(np.sqrt((high[inner]**2).mean())) if inner.any() else 0
    if rms<.012:
        return {'label':'Liso', 'basis':'Poca variación de detalle en el interior de la prenda', 'estimated':True}
    x=periodicity(high,inner,1);y=periodicity(high,inner,0)
    if x and y:
        label='Cuadros o rejilla'
    elif x:
        label='Rayas verticales'
    elif y:
        label='Rayas horizontales'
    else:
        label='Irregular o sin patrón concluyente'
    return {'label':label,'basis':'Periodicidad del color en el interior de la máscara; pliegues y sombras pueden alterar la lectura','estimated':True}


def characterize(norm, features, source_rgb=None, source_mask=None):
    ratio=float(features.get('ratio_aspecto',1))
    silhouette='Alargada' if ratio>=1.8 else ('Ancha' if ratio<=.9 else 'Intermedia')
    upper=np.mean([features.get('anchura_1',0),features.get('anchura_2',0)])
    lower=np.mean([features.get('anchura_5',0),features.get('anchura_6',0)])
    cut='No concluyente'
    if upper>.05 and lower>.05:
        quotient=lower/upper
        cut='Recto' if .8<=quotient<=1.2 else ('Ensanchado hacia abajo' if quotient>1.2 else 'Estrechado hacia abajo')
    light=float(features.get('claridad_media',features.get('color_L',50)))
    texture=analyze_texture(source_rgb if source_rgb is not None else norm.rgb,
                            source_mask if source_mask is not None else norm.mascara)
    pattern_info=pattern(norm,features)
    if texture['reliable']:
        pattern_info={'label':texture['pattern_label'],'basis':texture['basis'],'estimated':True}
    pattern_group = pattern_info['label']
    if pattern_group == 'Sin patrón repetitivo visible' and texture['label'] == 'Sin microtextura visible':
        pattern_group = 'Liso'
    color = features.get('color_nombre', '')
    return {'color':color, 'color_family':COLOR_FAMILIES.get(color, color),
            'texture':texture['label'], 'texture_analysis':texture,
            'shape':silhouette, 'silhouette':silhouette, 'cut':cut,
            'symmetry':'Aproximadamente simétrica' if features.get('simetria',0)>=.7 else 'Asimétrica o inclinada',
            'brightness':'Clara' if light>=70 else ('Oscura' if light<35 else 'Media'),
            'color_intensity':'Viva' if features.get('croma_medio',0)>=30 else 'Apagada',
            'color_composition':'Un color predominante'
            if features.get('n_colores_cromaticos', features.get('n_colores', 1)) <= 1 else 'Varios colores',
            'pattern':pattern_info, 'pattern_group':pattern_group, 'estimated':True}
