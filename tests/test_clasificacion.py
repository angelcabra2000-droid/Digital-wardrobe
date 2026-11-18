import json
from types import SimpleNamespace
import numpy as np
from closet import clasificacion as c
from closet.atributos import characterize, pattern


def test_neighbors_use_features():
    records=[{'id':'a','category':'camisas','features':{'ratio_aspecto':.7}},
             {'id':'b','category':'pantalones','features':{'ratio_aspecto':2.8}},
             {'id':'c','category':'chaquetas','features':{'ratio_aspecto':1.4}}]
    assert c.nearest({'ratio_aspecto':2.7},records,k=1)['category']=='pantalones'
    assert c.nearest({},[])['category'] is None


def test_knn_leaves_category_empty_for_distant_or_divided_neighbors():
    records = [{'id': name, 'category': name, 'features': {'ratio_aspecto': ratio}}
               for name, ratio in [('camisas', 1), ('chaquetas', 2)]]
    distant = c.nearest({'ratio_aspecto': 10}, records, k=1, blocks=(0,), max_distance=.8)
    assert distant['category'] is None and distant['review']
    divided = c.nearest({'ratio_aspecto': 1.5}, records, k=2, blocks=(0,), max_distance=.8)
    assert divided['category'] is None and divided['support'] == .5


def test_silhouette_distinguishes_straps_from_pant_legs():
    mask = np.zeros((120, 100), dtype=bool)
    mask[:32, 15:23] = True
    mask[:32, 76:84] = True
    mask[30:, 6:94] = True
    knn = {'category':'pantalones'}
    refined = c.refine_silhouette(knn, mask)
    assert (refined['category'], refined['subtype']) == ('camisas', 'Esqueleto')
    assert c.refine_silhouette(knn, mask[::-1]) == knn
    sleeves = mask.copy()
    sleeves[:32, 10:90] = True
    assert c.refine_silhouette(knn, sleeves) == knn


def test_classification_excludes_query_group(tmp_path,monkeypatch):
    model=tmp_path/'model.json'
    model.write_text(json.dumps({'records':[
        {'id':'self','group':'same','category':'camisas','features':{'ratio_aspecto':3}},
        {'id':'copy','group':'same','category':'camisas','features':{'ratio_aspecto':3}},
        {'id':'other','group':'other','category':'pantalones','features':{'ratio_aspecto':3}}]}))
    monkeypatch.setattr(c,'MODEL',model)
    result=c.classify({'ratio_aspecto':3},'self','same')
    assert result['category']=='pantalones'
    assert result['neighbors']==[{'id':'other','category':'pantalones'}]


def norm_for(rgb):
    return SimpleNamespace(rgb=rgb,mascara=np.ones(rgb.shape[:2],bool),nucleo=np.ones(rgb.shape[:2],bool))


def test_plain_and_geometric_descriptions():
    norm=norm_for(np.full((128,128,3),80,np.uint8))
    f={'ratio_aspecto':2.3,'anchura_1':.5,'anchura_2':.5,'anchura_5':.5,'anchura_6':.5,
       'simetria':.9,'claridad_media':20,'n_colores':1,'estampado':'liso'}
    a=characterize(norm,f)
    assert a['silhouette']=='Alargada' and a['cut']=='Recto'
    assert a['brightness']=='Oscura' and a['pattern']['label']=='Sin patrón repetitivo visible'


def test_periodic_patterns():
    y,x=np.indices((128,128))
    vertical=np.repeat(np.where((x//8)%2,210,40)[...,None],3,axis=2).astype(np.uint8)
    horizontal=np.repeat(np.where((y//8)%2,210,40)[...,None],3,axis=2).astype(np.uint8)
    checks=np.repeat(np.where(((x//8)+(y//8))%2,210,40)[...,None],3,axis=2).astype(np.uint8)
    f={'estampado':'con_textura'}
    assert pattern(norm_for(vertical),f)['label']=='Rayas verticales'
    assert pattern(norm_for(horizontal),f)['label']=='Rayas horizontales'
    assert pattern(norm_for(checks),f)['label']=='Cuadros o rejilla'


def test_train_ignores_test_and_excluded(tmp_path,monkeypatch):
    (tmp_path/'datos').mkdir();(tmp_path/'Aplicacion base de datos').mkdir()
    (tmp_path/'datos/dataset.csv').write_text('id,ruta,categoria,incluir,split,grupo\na,a.jpg,camisas,si,entrenamiento,a\nb,b.jpg,chaquetas,si,entrenamiento,b\nc,c.jpg,pantalones,si,entrenamiento,c\nd,test.jpg,camisas,si,prueba,d\ne,excluded.jpg,pantalones,no,entrenamiento,e\n')
    seen=[]
    def processing(path,**kwargs):
        seen.append(path.name)
        return SimpleNamespace(normalizada=path)
    monkeypatch.setattr(c,'procesar_foto',processing)
    monkeypatch.setattr(c,'extraer_caracteristicas',lambda norm:SimpleNamespace(como_dict=lambda:{'ratio_aspecto':1}))
    model=c.train(tmp_path,tmp_path/'model.json')
    assert seen==['a.jpg','b.jpg','c.jpg']
    assert model['validation']['test_set_used'] is False
