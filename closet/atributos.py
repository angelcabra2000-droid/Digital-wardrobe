"""Descriptores geométricos y de patrón; estimaciones visuales, no materiales textiles."""
import numpy as np
from .segmentacion import filtro_caja
from .texturas import analyze_texture


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
    return {'color':features.get('color_nombre',''), 'texture':texture['label'], 'texture_analysis':texture,
            'shape':silhouette, 'silhouette':silhouette, 'cut':cut,
            'symmetry':'Aproximadamente simétrica' if features.get('simetria',0)>=.7 else 'Asimétrica o inclinada',
            'brightness':'Clara' if light>=70 else ('Oscura' if light<35 else 'Media'),
            'color_intensity':'Viva' if features.get('croma_medio',0)>=30 else 'Apagada',
            'color_composition':'Un color predominante' if features.get('n_colores',1)<=1 else 'Varios colores',
            'pattern':pattern_info, 'estimated':True}
