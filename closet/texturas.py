"""Microtextura y repetición visibles, sin inferir composición o tacto de la tela.

Trabaja a resolución de entrada, no sobre la prenda normalizada a 256 px.
Separa variación rápida (trama/detalle) de cambios lentos (luz/pliegues).
"""
import numpy as np
from PIL import Image
from .segmentacion import filtro_caja, erosion


def _periodic(signal, mask, axis):
    correlations=[]
    length=signal.shape[axis]
    for shift in range(2,min(49,length//3)):
        if axis==1:
            a,b=signal[:,:-shift],signal[:,shift:];valid=mask[:,:-shift]&mask[:,shift:]
        else:
            a,b=signal[:-shift],signal[shift:];valid=mask[:-shift]&mask[shift:]
        if valid.sum()<100:
            correlations.append(0.);continue
        aa,bb=a[valid],b[valid]
        aa=aa-aa.mean();bb=bb-bb.mean()
        den=float(np.sqrt((aa*aa).sum()*(bb*bb).sum()))
        correlations.append(float((aa*bb).sum()/den) if den>1e-8 else 0.)
    peaks=[(v,i+2) for i,v in enumerate(correlations[1:-1],start=1)
           if v>correlations[i-1] and v>correlations[i+1]]
    if not peaks:return {'present':False,'strength':0.,'period_px':None}
    strength,period=max(peaks)
    return {'present':bool(strength>.38 and min(correlations)<-.08),
            'strength':round(strength,4),'period_px':period}


def analyze_texture(rgb, mask):
    rgb=np.asarray(rgb,dtype=np.uint8);mask=np.asarray(mask,bool)
    if rgb.shape[:2]!=mask.shape or rgb.ndim!=3:
        raise ValueError('Imagen y máscara de textura no coinciden')
    ys,xs=np.nonzero(mask)
    empty={'label':'No concluyente','estimated':True,'reliable':False,
           'pattern_label':'No concluyente','basis':'Falta detalle interior suficiente en la fotografía.',
           'metrics':{},'method':'contraste multiescala y patrones binarios locales'}
    if len(xs)<120 or min(int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1))<24:
        return empty
    # Recorte sin interpolar hacia arriba: no se inventan hilos de una foto pequeña.
    rgb=rgb[ys.min():ys.max()+1,xs.min():xs.max()+1]
    mask=mask[ys.min():ys.max()+1,xs.min():xs.max()+1]
    if max(mask.shape)>768:
        h,w=mask.shape;size=(round(w*768/max(h,w)),round(h*768/max(h,w)))
        rgb=np.asarray(Image.fromarray(rgb).resize(size,Image.Resampling.LANCZOS))
        mask=np.asarray(Image.fromarray(mask.astype('uint8')).resize(size,Image.Resampling.NEAREST),bool)
    gray=rgb.astype(np.float64)@np.array([.2126,.7152,.0722])/255
    m=mask.astype(float)
    averages=[filtro_caja(gray*m,r)/np.maximum(filtro_caja(m,r),1e-6) for r in (1,3,7)]
    inside=erosion(mask,2)&(filtro_caja(m,7)>.95)
    if inside.sum()<120:return empty
    fine=gray-averages[0];middle=averages[0]-averages[1];coarse=averages[1]-averages[2]
    # Medición de microvariación sobre regiones, para que una costura aislada no domine.
    patch_energy=[]
    for y in range(0,gray.shape[0]-7,8):
        for x in range(0,gray.shape[1]-7,8):
            valid=inside[y:y+8,x:x+8]
            if valid.sum()>=48:
                patch_energy.append(float(np.sqrt((fine[y:y+8,x:x+8][valid]**2).mean())))
    rms_f=float(np.sqrt((fine[inside]**2).mean()))
    rms_m=float(np.sqrt((middle[inside]**2).mean()))
    rms_c=float(np.sqrt((coarse[inside]**2).mean()))
    density=float((np.abs(fine[inside])>.012).mean())
    patches=float(np.mean(np.asarray(patch_energy)>.012)) if patch_energy else density
    # LBP con tolerancia fotométrica: igualdad/ruido pequeño no crea relieve artificial.
    bits=[];valid=erosion(inside,1)
    for dy,dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
        bits.append(np.roll(gray,(dy,dx),(0,1))>gray+.008)
    bits=np.stack(bits)
    transitions=(bits!=np.roll(bits,1,axis=0)).sum(0)
    codes=np.where(transitions<=2,bits.sum(0),9)
    hist=np.bincount(codes[valid].astype(int),minlength=10).astype(float)
    hist/=max(hist.sum(),1);nonzero=hist[hist>0]
    entropy=float(-(nonzero*np.log2(nonzero)).sum())
    signal=(gray-averages[2])*inside
    px=_periodic(signal,inside,1);py=_periodic(signal,inside,0)
    textured=bool(density>.08 and patches>.15 and rms_f>.008)
    # Nunca denominar material (algodón, lana, encaje...) solo por contraste.
    if not textured:
        label='Sin microtextura visible'
    elif rms_f>=.055:
        label='Textura marcada / detalle contrastado'
    elif px['present'] and py['present']:
        label='Microtextura repetitiva'
    elif rms_f>=rms_m*.9:
        label='Textura fina visible'
    else:
        label='Textura visual irregular'
    fine_pattern=bool(textured and (px['present'] or py['present']) and rms_f>=rms_m*1.1
                      and min([v['period_px'] for v in (px,py) if v['present']])<=8)
    if not textured and max(rms_f,rms_m)<.006:pattern='Sin patrón repetitivo visible'
    elif fine_pattern:pattern='Trama fina repetitiva'
    elif px['present'] and py['present']:pattern='Cuadros o rejilla'
    elif px['present']:pattern='Rayas verticales'
    elif py['present']:pattern='Rayas horizontales'
    elif textured:pattern='Detalle irregular / patrón no concluyente'
    else:pattern='Sin patrón repetitivo visible'
    metrics={'fine_rms':round(rms_f,5),'medium_rms':round(rms_m,5),'coarse_rms':round(rms_c,5),
             'detail_density':round(density,4),'textured_patch_fraction':round(patches,4),
             'lbp_entropy':round(entropy,4),'periodicity_x':px,'periodicity_y':py,
             'source_width':int(rgb.shape[1]),'source_height':int(rgb.shape[0]),'interior_pixels':int(inside.sum())}
    return {'label':label,'estimated':True,'reliable':True,'pattern_label':pattern,
            'basis':'Detalle local y repetición medidos dentro de la prenda a resolución de entrada; no determina material ni tacto.',
            'metrics':metrics,'method':'contraste multiescala y patrones binarios locales'}
