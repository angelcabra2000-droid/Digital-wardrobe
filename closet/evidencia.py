"""Genera un informe HTML autocontenido del procesamiento y la clasificación."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
from html import escape
from io import BytesIO
import json
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np

from .caracteristicas import extraer_caracteristicas, luminancia
from .clasificacion import BLOCKS, BLOCK_NAMES, MODEL, ROOT, SUBTYPES, nearest, refine_silhouette
from .alertas import fondo_impide_segmentar
from .config import PreprocesoConfig
from .dataset import Registro, nombre_en_muestra
from .evaluacion import cargar_mascara, metricas, reducir_mascara
from .pipeline import procesar_foto
from .segmentacion import imagen_clusters, recortar_fondo


def histogram(original, processed):
    figure = Figure(figsize=(8, 3), layout='constrained')
    axis = figure.subplots()
    for rgb, label in ((original, 'Original'), (processed, 'Preprocesada')):
        axis.hist(luminancia(rgb.astype(float)).ravel(), bins=32, range=(0, 256),
                  histtype='step', weights=np.full(rgb.shape[:2], 100 / np.prod(rgb.shape[:2])).ravel(),
                  linewidth=2, label=label)
    axis.set(xlabel='Luminancia (0–255)', ylabel='% de píxeles por intervalo', xlim=(0, 256))
    axis.legend()
    return figure


def inline_plot(figure, alt):
    buffer = BytesIO()
    figure.savefig(buffer, format='png', dpi=125)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'<img alt="{escape(alt)}" src="data:image/png;base64,{encoded}">'


def feature_plots(records, labels):
    figure = Figure(figsize=(11, 3.6), layout='constrained')
    for axis, (key, title) in zip(figure.subplots(1, 3), (
        ('ratio_aspecto', 'Relación alto/ancho'), ('color_L', 'Claridad L*'),
        ('har_contraste', 'Contraste Haralick'),
    )):
        axis.boxplot([[r['features'][key] for r in records if r['category'] == label] for label in labels])
        axis.set_xticks(range(1, len(labels)+1), labels, rotation=20)
        axis.set_title(title)
        axis.grid(axis='y', alpha=.2)
    scatter = Figure(figsize=(7, 4), layout='constrained')
    axis = scatter.subplots()
    for label in labels:
        vectors = [r['features'] for r in records if r['category'] == label]
        axis.scatter([f['ratio_aspecto'] for f in vectors], [f['solidez'] for f in vectors],
                     alpha=.7, label=label)
    axis.set(xlabel='Relación alto/ancho', ylabel='Solidez')
    axis.legend()
    return inline_plot(figure, 'Cajas y bigotes por categoría') + inline_plot(scatter, 'Dispersión de alto/ancho y solidez')


def example_plot(result, truth):
    pre, seg, norm = result.preproceso, result.segmentacion, result.normalizada
    figure = Figure(figsize=(11, 5.5), layout='constrained')
    stages = [('Original', pre.original), ('Preprocesada', pre.normalizada),
              ('Clusters', imagen_clusters(seg, pre.original)), ('Máscara', seg.mascara),
              ('Sin fondo', recortar_fondo(pre.original, seg.mascara)),
              ('Normalizada', recortar_fondo(norm.rgb, norm.mascara))]
    if truth is not None:
        stages.append(('Máscara manual', truth))
    axes = figure.subplots(2, 4).ravel()
    for axis in axes:
        axis.axis('off')
    for axis, (title, image) in zip(axes, stages):
        axis.imshow(image, cmap='gray', vmin=0, vmax=1 if image.ndim == 2 else 255)
        axis.set_title(title)
    return inline_plot(figure, 'Etapas del procesamiento') + inline_plot(
        histogram(pre.original, pre.normalizada), 'Histograma de luminancia antes y después')


def table(headers, rows):
    heading = ''.join(f'<th scope="col">{escape(str(value))}</th>' for value in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(str(value))}</td>' for value in row) + '</tr>' for row in rows)
    return f'<table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table>'


def generate(output):
    model = json.loads(MODEL.read_text(encoding='utf-8'))
    records, settings = model['records'], model['settings']
    with (ROOT / 'datos/dataset.csv').open(encoding='utf-8-sig', newline='') as stream:
        catalog = list(csv.DictReader(stream))
    test = [r for r in catalog if r['split'] == 'prueba' and r['incluir'] == 'si']
    if not test:
        raise ValueError('El catálogo no contiene fotos incluidas de prueba')
    test.sort(key=lambda row: row['muestra_mascara'] != 'si')
    if {r['grupo'] or r['id'] for r in test} & {r['group'] for r in records}:
        raise ValueError('Hay grupos compartidos entre entrenamiento y prueba')
    labels = sorted({r['category'] for r in records})
    blocks = lambda indices: ' + '.join(BLOCK_NAMES[index] for index in indices)
    report = [
        '<h1>Closet Gossip · Evidencia experimental</h1>',
        '<p>Una foto por prenda. El informe mide la calidad del recorte y los aciertos del KNN con métricas propias.</p>',
        '<p><a href="/estudio">Estudio interactivo</a> · <a href="/evidencia/segmentacion">Comparación de segmentadores</a> · <a href="#clasificacion">Clasificación KNN</a> · <a href="#correcciones">Aprendizaje por correcciones</a></p>',
        '<details open><summary><strong>Cómo funciona la aplicación, paso a paso</strong></summary><ol>',
        '<li><strong>Entrada.</strong> La app recibe una foto de cámara, archivo o catálogo. Aplica la orientación EXIF y el giro manual, y reduce la imagen al tamaño de análisis.</li>',
        '<li><strong>Preprocesamiento.</strong> El balance de blancos compensa dominantes de iluminación. El suavizado atenúa ruido y el ajuste de contraste amplía diferencias de claridad. El histograma compara el porcentaje de píxeles por intervalo de luminancia antes y después.</li>',
        '<li><strong>Segmentación.</strong> K-means agrupa píxeles por color Lab y textura local. El análisis de regiones y la morfología producen una máscara. U²-Net aporta una estimación independiente de qué píxeles son ropa. El híbrido aprovecha la coincidencia de las máscaras para ajustar el contorno.</li>',
        '<li><strong>Calidad y normalización.</strong> La app revisa área, contacto con bordes, contraste y variación del fondo. Una máscara válida delimita un recorte centrado en un lienzo de 256 × 256 píxeles. El color se mide con los píxeles originales dentro de la prenda.</li>',
        '<li><strong>Características.</strong> Se miden proporciones, perfiles de anchura y tramos de la silueta, color Lab y textura Haralick. La periodicidad permite distinguir patrones como rayas y cuadros.</li>',
        '<li><strong>Ficha.</strong> El KNN sugiere el tipo y, si encuentra evidencia, un subtipo. El usuario revisa la ficha, corrige los campos necesarios y guarda la prenda en su clóset.</li></ol></details>',
        '<h2>1. Catálogo y descriptores</h2>',
        table(('Categoría', 'Entrenamiento incluido', 'Prueba incluida', 'Excluidas', 'Vectores entrenados'),
              [(label, sum(r['split'] == 'entrenamiento' and r['incluir'] == 'si' for r in rows),
                sum(r['split'] == 'prueba' and r['incluir'] == 'si' for r in rows),
                sum(r['incluir'] != 'si' for r in rows), sum(r['category'] == label for r in records))
               for label in labels for rows in [[r for r in catalog if r['categoria'] == label]]]),
        '<p>Las cajas muestran los cuartiles y la mediana. Los bigotes alcanzan las observaciones dentro '
        'de 1,5 veces el rango intercuartílico. Cada punto de dispersión es una foto de entrenamiento.</p>',
        feature_plots(records, labels),
        '<h2 id="clasificacion">2. Clasificación KNN y selección de características</h2>',
        '<p>K-means agrupa píxeles y KNN compara prendas. Para clasificar, KNN estandariza los descriptores con las estadísticas de referencia, ordena los ejemplos por distancia y da más peso a los votos cercanos. El KNN de subcategoría busca dentro de la familia sugerida y requiere etiquetas con ejemplos de grupos independientes.</p>',
        '<p>La exactitud equilibrada promedia el porcentaje de aciertos de cada clase. Así, la categoría con más fotos tiene el mismo peso que las demás. La selección interna compara 28 combinaciones antes de aplicar el rechazo por incertidumbre. La evaluación reservada de la sección 4 aplica el rechazo.</p>',
        '<p>En cada iteración se deja fuera un grupo completo de fotos relacionadas, se ajusta la escala '
        'con los grupos restantes y se vota por distancia. Se maximiza la media del recall de las tres clases. '
        'Los empates favorecen menos variables y un k menor.</p>',
        table(('Bloque', 'Cantidad', 'Variables'),
              [(name, len(keys), ', '.join(keys)) for name, keys in zip(BLOCK_NAMES, BLOCKS)]),
        table(('Bloques', 'Variables', 'k', 'Exactitud equilibrada', 'Exactitud'),
              [(blocks(r['blocks']), r['features'], r['k'], f"{r['balanced_accuracy']:.1%}", f"{r['accuracy']:.1%}")
               for r in model['experiments']]),
        f"<p>Configuración elegida <strong>{blocks(settings['blocks'])} · "
        f"{model['validation']['feature_count']} variables · k = {settings['k']}</strong>. "
        f"Se deja la categoría vacía cuando el vecino más cercano supera la distancia "
        f"{settings['max_distance']:.3f} (percentil 95 en entrenamiento) o el voto ponderado ganador baja del 70 %. "
        "El voto expresa acuerdo entre vecinos y la distancia mide cercanía en el espacio de descriptores. "
        "Si la máscara muestra dos tirantes finos sobre un torso continuo, la geometría permite sugerir Esqueleto. "
        "Se bloquea la ficha si la máscara ocupa tramos extensos de dos bordes, si toca un borde con poco contraste o fondo irregular, "
        "o si el fondo complejo coincide con poco contraste o área anómala. La máscara se conserva para inspección.</p>",
        '<p>La distancia combina el error cuadrático medio de cada bloque después de estandarizar. '
        'Los pesos iniciales fueron 80 % para forma, 8 % para color y 12 % para textura. '
        'Se renormalizan al omitir bloques. Se comparan bloques completos de variables.</p>',
        '<h2>3. Procesamiento y ejemplos de prueba</h2>',
        '<p>Se corrige la orientación y la iluminación, se agrupan los píxeles por color y textura y se limpia '
        'la máscara con operaciones morfológicas. El recorte normalizado permite comparar siluetas. '
        'El color se mide en Lab sobre el interior del recorte original. La repetición en ambos ejes '
        'permite distinguir cuadros y rayas. El detalle local ayuda a estimar la textura visible.</p>',
    ]
    results, examples = [], set()
    output.mkdir(parents=True, exist_ok=True)
    for row in test:
        result = procesar_foto(ROOT / 'Aplicacion base de datos' / row['ruta'],
                               cfg_pre=PreprocesoConfig(lado_max=model['training_size']))
        label = row['categoria']
        seg = result.segmentacion
        sample_name = nombre_en_muestra(Registro(ruta=row['ruta'], categoria=label))
        manual_path = ROOT / 'muestra' / (Path(sample_name).stem + '_gt.png')
        truth = reducir_mascara(cargar_mascara(manual_path), *seg.mascara.shape) if manual_path.is_file() else None
        prediction = {}
        blocked = fondo_impide_segmentar(result.alertas)
        if result.normalizada is not None and not blocked:
            features = extraer_caracteristicas(result.normalizada).como_dict()
            prediction = refine_silhouette(nearest(features, records, **settings), result.normalizada.mascara)
        knn = prediction.get('knn_prediction', prediction)
        iou, precision, recall, f1 = metricas(seg.mascara, truth) if truth is not None else (None,)*4
        results.append({'id':row['id'], 'real':label, 'predicha':prediction.get('category'),
                        'segmentada':bool(seg.encontrada), 'bloqueada_por_fondo':blocked,
                        'metodo':prediction.get('method'), 'voto_knn':knn.get('support'),
                        'distancia':knn.get('distance'), 'iou':iou, 'precision_mascara':precision,
                        'recall_mascara':recall, 'f1_mascara':f1})
        if label not in examples and result.normalizada is not None:
            report.append(f"<h3>{escape(row['id'])} · {escape(label)}</h3>")
            report.append(example_plot(result, truth))
            examples.add(label)
        print(f'{len(results)}/{len(test)}: {row["id"]}', flush=True)
    with (output / 'predicciones.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    counts = Counter((r['real'], r['predicha']) for r in results)
    accepted = sum(bool(r['predicha']) for r in results)
    correct = sum(counts[label, label] for label in labels)
    ious = [r['iou'] for r in results if r['iou'] is not None]
    subtype_counts = Counter((r['category'], r.get('subtype')) for r in records)
    report += [
        '<h2>4. Evaluación en fotos reservadas</h2>',
        '<p>La revisión de capturas usó dos ejemplos de entrenamiento para comprobar la regla de tirantes '
        'y el filtro de fondo. También se inspeccionó pan_006, que pertenece a prueba, para corregir la composición '
        'cromática. Su conteo ajustado de colores se usa en la ficha y queda fuera de las entradas del KNN.</p>',
        f'<p>{correct}/{len(results)} aciertos sobre todas las fotos y {accepted}/{len(results)} '
        f'con categoría asignada. Entre las asignadas hubo {correct}/{accepted} aciertos. '
        f'El filtro de fondo bloqueó {sum(r["bloqueada_por_fondo"] for r in results)} fotos.</p>',
        table(('Real ↓ / Predicha →', *labels, 'Vacía'),
              [(label, *(counts[label, guess] for guess in [*labels, None])) for label in labels]),
        f'<p>IoU medio de segmentación: {np.mean(ious):.3f} sobre {len(ious)} fotos de prueba con máscara manual.</p>'
        if ious else '<p>Sin máscaras manuales disponibles en prueba.</p>',
        '<h2 id="correcciones">5. Aprendizaje por correcciones</h2>',
        '<ol><li>El usuario cambia el tipo o la subcategoría y guarda la ficha. Si el recorte pasó sin alertas, SQLite conserva el vector y la etiqueta corregida para esa foto y ese perfil.</li>',
        '<li>En el siguiente análisis el KNN consulta hasta 60 correcciones recientes. Solo sugiere una corrección personal si el vecino más cercano tiene distancia de 0,18 o menos y el voto ganador alcanza el 70 %.</li>',
        '<li>Para demostrarlo, repite la foto con el mismo perfil después de corregirla. Luego prueba otra toma de la misma prenda para comprobar hasta dónde se transfiere la sugerencia.</li></ol>',
        '<p>El aprendizaje funciona por memoria de ejemplos. El catálogo fijo se utiliza para calcular los resultados académicos. El análisis del perfil consulta su propia memoria y permite revisar la sugerencia personalizada.</p>',
        '<h2>6. Alcance de las etiquetas</h2>',
        f"<p>Subcategorías anotadas en el entrenamiento: {sum(bool(r.get('subtype')) for r in records)}. "
        "Estas anotaciones se hicieron a partir de la apariencia en la foto. Un subtipo necesita al menos "
        "dos grupos independientes para generar sugerencias. La prueba reservada evalúa la categoría general. "
        "La textura, el color y el subtipo requieren etiquetas de prueba propias para medir su exactitud.</p>",
        table(('Categoría', 'Subcategoría', 'Fotos anotadas', 'Sugerencia automática'),
              [(category, subtype, subtype_counts[category, subtype],
                'Disponible' if subtype in model['subtypes'].get(category, {}).get('labels', []) else 'Solo selección manual')
               for category, options in SUBTYPES.items() for subtype in options]),
        table(('Categoría', 'Bloques del KNN de subcategoría', 'Variables', 'k', 'Exactitud equilibrada interna'),
              [(category, blocks(submodel['settings']['blocks']), submodel['validation']['features'],
                submodel['settings']['k'], f"{submodel['validation']['balanced_accuracy']:.1%}")
               for category, submodel in model['subtypes'].items()]),
    ]
    document = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Evidencia · Closet Gossip</title><style>body{max-width:980px;margin:2rem auto;padding:0 1rem;font:16px/1.55 system-ui;color:#243d37}img{max-width:100%;height:auto;display:block;margin:1rem 0}table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border-bottom:1px solid #d6e4df;padding:.4rem;text-align:left}th{background:#edf4ef}h2{margin-top:2rem}</style><main>'
    (output / 'informe.html').write_text(document + '\n'.join(report) + '</main></html>', encoding='utf-8')


def compare_segmentation(output):
    """Compara las tres máscaras con la misma foto y el mismo KNN."""
    from .dataset import cargar_csv
    from .normalizacion import normalizar_prenda
    from .alertas import revisar_segmentacion
    from .red_neuronal import segmentation_ready

    if not segmentation_ready():
        raise RuntimeError('Instala onnxruntime y ejecuta python -m closet.preparar_cnn --solo-segmentacion')
    model = json.loads(MODEL.read_text(encoding='utf-8'))
    names = ('K-means', 'U²-Net', 'Híbrido')
    catalog = cargar_csv(ROOT / 'datos/dataset.csv')
    test = [r for r in catalog if r.incluir and r.split == 'prueba']
    if {r.grupo or r.id for r in test} & {r['group'] for r in model['records']}:
        raise ValueError('Hay grupos compartidos entre entrenamiento y prueba')
    rows, examples = [], []
    for index, row in enumerate(test, 1):
        path = ROOT / 'muestra' / (Path(nombre_en_muestra(row)).stem + '_gt.png')
        result = procesar_foto(ROOT / 'Aplicacion base de datos' / row.ruta,
                              cfg_pre=PreprocesoConfig(lado_max=model['training_size']), motor='hibrido')
        original = result.preproceso.original
        truth = reducir_mascara(cargar_mascara(path), *original.shape[:2]) if row.muestra_mascara and path.is_file() else None
        masks = (result.mascaras['mascara_kmeans'], result.mascaras['mascara_u2net'], result.segmentacion.mascara)
        measures = []
        for name, mask in zip(names, masks):
            iou, _, _, f1 = metricas(mask, truth) if truth is not None else (None,)*4
            prediction = None
            if mask.any() and not fondo_impide_segmentar(revisar_segmentacion(original, mask)[0]):
                norm = normalizar_prenda(original, mask)
                features = extraer_caracteristicas(norm).como_dict()
                guess = nearest(features, model['records'], **model['settings'])
                prediction = refine_silhouette(guess, norm.mascara)['category']
            rows.append({'id':row.id,'real':row.categoria,'segmentador':name,'iou':iou,'f1':f1,'predicha':prediction})
            measures.append(iou)
        if truth is not None:
            examples.append((measures[0], row.id, original, truth, masks, measures))
        print(f'{index}/{len(test)} {row.id}: ' + (' / '.join(f'{value:.3f}' for value in measures)
              if truth is not None else 'clasificación sin máscara de referencia'), flush=True)
    if not rows:
        raise ValueError('No hay fotos reservadas para la evaluación')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'comparacion_segmentacion.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    distributions = []
    outcomes = []
    for name in names:
        subset = [row for row in rows if row['segmentador'] == name]
        ious = [row['iou'] for row in subset if row['iou'] is not None]
        distributions.append(ious)
        correct = sum(row['real'] == row['predicha'] for row in subset)
        empty = sum(not row['predicha'] for row in subset)
        wrong = len(subset) - empty - correct
        outcomes.append((correct, wrong, empty))
        summaries.append((name, len(subset), correct, wrong, empty, len(ious),
                          f'{np.mean(ious):.3f}' if ious else '—'))
    figure = Figure(figsize=(10, 4), layout='constrained')
    classification, segmentation = figure.subplots(1, 2)
    bottom = np.zeros(len(names))
    for values, label, color in zip(zip(*outcomes), ('Correcta', 'Equivocada', 'Vacía'),
                                     ('#26745f', '#ae5638', '#8897a3')):
        classification.bar(names, values, bottom=bottom, label=label, color=color)
        bottom += values
    classification.set(ylabel='Fotos de prueba', title='Resultado del mismo KNN')
    classification.legend()
    if examples:
        segmentation.boxplot(distributions)
        segmentation.set_xticks(range(1, 4), names)
        segmentation.set(ylabel='IoU con máscara manual', ylim=(0, 1.02), title='Calidad del recorte anotado')
    else:
        segmentation.text(.5, .5, 'Sin máscaras manuales de prueba', ha='center', va='center')
        segmentation.axis('off')
    parts = ['<!doctype html><html lang="es"><meta charset="utf-8"><title>Comparación de segmentación</title>'
             '<style>body{margin:2rem auto;max-width:980px;padding:0 1rem;font:16px system-ui;color:#243d37}'
             'img{max-width:100%;height:auto}th,td{padding:.5rem;border-bottom:1px solid #d6e4df}</style>'
             '<h1>Comparación de segmentadores</h1>'
             '<p><a href="/evidencia">Volver a Evidencia</a> · <a href="/estudio">Probar en el Estudio</a></p>'
             '<h2>Qué aporta cada método</h2><p>K-means usa color y textura locales. U²-Net aporta una región de ropa '
             'aprendida de otras imágenes. Si el IoU entre ambas máscaras alcanza 0,45, el híbrido conserva el núcleo '
             'neuronal y deja que K-means ajuste una banda estrecha del borde. Si el ajuste cambia más del 15 % del área '
             'o hay poco acuerdo, se conserva la máscara neuronal. Si la red produce una máscara vacía se usa K-means.</p>'
             '<h2>Diseño de la comparación</h2><p>Se comparan las mismas fotos con la misma resolución y el mismo KNN. '
             'El IoU divide los píxeles compartidos por las dos máscaras entre todos los píxeles cubiertos por cualquiera '
             'de ellas. El diagrama de cajas muestra mediana, cuartiles y dispersión. En los ejemplos de abajo, '
             'compara fondo adherido, bordes y piezas delgadas antes de interpretar la cifra global.</p>'
             f'<p>La clasificación se compara en las {len(test)} fotos reservadas de prueba. '
             f'El IoU se calcula en las {len(examples)} fotos de prueba con máscara manual disponible. '
             'Las correcciones personales quedan fuera de la evaluación.</p>',
             table(('Segmentador','Fotos de prueba','Aciertos','Errores','Vacías','Con máscara manual','IoU medio'), summaries),
             inline_plot(figure, 'Resultados de clasificación e IoU por segmentador'),
             '<p>Las barras usan todas las fotos reservadas. Las cajas de IoU usan el subconjunto anotado. '
             'Una categoría vacía indica que el recorte se bloqueó o que el KNN aplicó el rechazo por incertidumbre.</p>']
    for _, ident, rgb, truth, masks, ious in sorted(examples, key=lambda example:example[0])[:3]:
        figure = Figure(figsize=(10, 3), layout='constrained')
        for axis, image, title in zip(figure.subplots(1, 5), (rgb, truth, *masks),
                                     ('Original','Máscara manual', *(f'{name}\nIoU {value:.2f}' for name, value in zip(names, ious)))):
            axis.imshow(image, cmap='gray', vmin=0, vmax=1 if image.ndim == 2 else 255)
            axis.set_title(title)
            axis.axis('off')
        parts += [f'<h2>{escape(ident)}</h2>', inline_plot(figure, 'Comparación de máscaras')]
    parts.append('</html>')
    (output / 'comparacion_segmentacion.html').write_text('\n'.join(parts), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida', type=Path, default=ROOT / 'resultados' / 'evidencia')
    parser.add_argument('--comparar-segmentadores', action='store_true')
    args = parser.parse_args()
    if args.comparar_segmentadores:
        compare_segmentation(args.salida)
    else:
        generate(args.salida)
