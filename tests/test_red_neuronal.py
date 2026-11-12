import numpy as np
import pytest
from PIL import Image
from closet import red_neuronal as cnn
from closet import pipeline
from closet.segmentacion import ResultadoSegmentacion


def test_labels_nearest_does_not_invent_classes():
    scores=np.zeros((1,4,2,2),np.float32)
    scores[0,1,:,0]=5;scores[0,3,:,1]=5
    labels=cnn.decode_labels(scores,(19,23))
    assert labels.shape==(23,19)
    assert set(np.unique(labels))=={1,3}
    # La pertenencia a ropa no fuerza camisa, pantalón o chaqueta por color.
    assert cnn.select_mask(labels).all()


def test_empty_and_assisted_regions():
    assert not cnn.select_mask(np.zeros((20,20),np.uint8)).any()
    labels=np.ones((20,20),np.uint8)
    upper=cnn.select_mask(labels,'upper');lower=cnn.select_mask(labels,'lower')
    assert not (upper&lower).any() and (upper|lower).all()
    with pytest.raises(ValueError):cnn.select_mask(labels,'unknown')


def test_missing_weights_refuse_random_network(tmp_path,monkeypatch):
    monkeypatch.setattr(cnn,'MODELS',tmp_path)
    assert cnn.status()['ready'] is False
    with pytest.raises(RuntimeError,match='no está instalada'):
        cnn.analyze(np.ones((50,50,3),np.uint8))


def test_cnn_pipeline_preserves_selected_mask_and_original_color(tmp_path,monkeypatch):
    a=np.full((90,80,3),225,np.uint8);a[15:75,20:60]=[25,90,140]
    photo=tmp_path/'shirt.png';Image.fromarray(a).save(photo)
    seen=[]
    def analyze(rgb,region):
        seen.append((rgb.copy(),region));m=np.zeros(rgb.shape[:2],bool);m[15:75,20:60]=True
        return ResultadoSegmentacion(m,m.astype('uint8'),2,1,{'engine':'cnn','classification':{'category':'camisas'}})
    monkeypatch.setattr(cnn,'analyze',analyze)
    monkeypatch.setattr(pipeline,'segmentar_imagen',lambda *_:pytest.fail('No debe ejecutar K-means en modo CNN'))
    result=pipeline.procesar_foto(photo,motor='cnn',region='upper')
    assert seen[0][1]=='upper'
    assert result.normalizada is not None
    assert result.segmentacion.diagnostico['classification']['category']=='camisas'
    assert result.segmentacion.mascara.sum()==2400
    assert np.array_equal(seen[0][0],result.preproceso.original)


def test_training_split_does_not_use_reserved_or_excluded(tmp_path):
    from closet.entrenar_cnn import load_rows,split_groups
    (tmp_path/'datos').mkdir()
    lines=['id,ruta,categoria,incluir,split,grupo']
    for category in cnn.CLASSES:
        for i in range(6):lines.append(f'{category}{i},photo.jpg,{category},si,entrenamiento,{category}{i//2}')
    lines+=['reserved,test.jpg,camisas,si,prueba,test','excluded,no.jpg,camisas,no,entrenamiento,no']
    (tmp_path/'datos/dataset.csv').write_text('\n'.join(lines))
    rows=load_rows(tmp_path);train,valid=split_groups(rows)
    assert len(rows)==18 and len(train)+len(valid)==18
    assert not {r['grupo'] for r in train}&{r['grupo'] for r in valid}
    assert {r['categoria'] for r in train}==set(cnn.CLASSES)
    assert {r['categoria'] for r in valid}==set(cnn.CLASSES)


def test_regions_separate_shirt_and_trousers_but_keep_same_height_fragments():
    labels = np.zeros((120, 80), dtype=np.uint8)
    labels[10:70, 10:70] = 1
    labels[70:110, 20:60] = 2
    top, bottom = cnn.separate_regions(labels)
    assert np.array_equal(top, labels == 1)
    assert np.array_equal(bottom, labels == 2)

    labels.fill(0)
    labels[10:70, 10:45] = 1
    labels[10:70, 45:70] = 2
    assert cnn.separate_regions(labels) is None
