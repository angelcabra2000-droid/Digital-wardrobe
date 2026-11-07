# Closet Digital

Segmenta y clasifica fotos de prendas usando **procesamiento de imágenes puro**: NumPy para todo el
procesamiento y Pillow solo para abrir y guardar archivos. K-means, los filtros, la morfología y
(más adelante) las características de Haralick están escritos a mano. Sin deep learning.

## Pipeline

| Paso | Módulo | Qué hace | Estado |
|---|---|---|---|
| 0 | `dataset`, `revision` | Inventariar la base, descartar fotos, dividir entrenamiento / prueba, elegir la muestra de máscaras | ✅ listo |
| 1 | `imagen_io`, `preprocesamiento`, `calidad` | Leer, unificar y revisar la foto | ✅ listo |
| 2 | `segmentacion`, `evaluacion` | K-means + regiones conectadas -> máscara de la prenda; medición con IoU | ✅ listo (IoU 0.92 en entrenamiento) |
| 3 | `normalizacion`, `alertas`, `pipeline` | Recortar, enderezar y escalar la prenda; alertas de recorte dudoso | ✅ listo |
| 4 | `caracteristicas` | Color, forma y textura (Haralick) | pendiente |
| 5 | `clasificacion` | Tipo de prenda y atributos | pendiente |
| 6-9 | `closet_db`, `lote`, `evaluacion`, interfaz | Closet, procesamiento por lotes, métricas, app | pendiente |

## Estructura

```
Closet_Digital/
├── closet/
│   ├── config.py            parámetros (DatasetConfig, PreprocesoConfig, CalidadConfig)
│   ├── dataset.py           paso 0: inventario, duplicados, división, muestra de máscaras
│   ├── revision.py          paso 0: hoja visual (miniaturas) para descartar fotos
│   ├── imagen_io.py         cargar (con orientación EXIF), guardar, listar imágenes
│   ├── preprocesamiento.py  tamaño, balance de blancos, suavizado, contraste
│   ├── calidad.py           nitidez, brillo, contraste -> advertencias
│   ├── segmentacion.py      paso 2: K-means a mano + regiones -> máscara de la prenda
│   ├── evaluacion.py        paso 2: IoU / precisión / recall contra máscaras hechas a mano
│   ├── normalizacion.py     paso 3: recortar, enderezar y escalar la prenda a un lienzo fijo
│   ├── alertas.py           paso 3: avisa cuando la segmentación probablemente está mal
│   ├── pipeline.py          encadena pasos 1-3 para una foto (procesar_foto)
│   └── cli.py               comandos: python -m closet ...
├── herramientas/
│   └── anotador_mascaras.html   dibujar a mano la máscara de las fotos de la muestra
├── tests/                   pruebas automáticas (pytest)
├── datos/                   (se crea sola) dataset.csv, exclusiones.txt, revision/
├── Aplicacion base de datos/   tus fotos (camisas, chaquetas, pantalones)
└── segmentar_camisa_kmeans.py  script anterior (ya migrado a closet/segmentacion.py; puedes borrarlo)
```

## Instalación

```
venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Paso 0: preparar la base (se hace una sola vez, antes de ajustar nada)

**1. Inventario y hoja de revisión**
```
python -m closet inventario "Aplicacion base de datos" --revision
```
Crea `datos/dataset.csv` (una fila por foto: id, categoría, tamaño, grupo, si se incluye, avisos y
a qué conjunto pertenece) y `datos/revision/revision.html`.

**2. Descartar las fotos que no sirven**
Abre `datos/revision/revision.html` con doble clic, haz clic en las fotos que quieras excluir
(personas con la prenda puesta, varias prendas, prenda cortada...) y pulsa *Descargar
exclusiones.txt*. Guárdalo en `datos/`. También puedes poner `incluir = no` en `dataset.csv`.

**3. Rehacer la división con las fotos que quedan** (solo esta vez)
```
python -m closet inventario "Aplicacion base de datos" --redividir
```

**4. Elegir las fotos cuya máscara se dibuja a mano**
```
python -m closet muestra "Aplicacion base de datos"
```
Copia unas 36 fotos (24 de entrenamiento y 12 de prueba, repartidas por categoría) a `muestra/`.
Abre `herramientas/anotador_mascaras.html` en Chrome o Edge, elige todas las fotos de `muestra/`,
dibuja el contorno de cada prenda (`H` para marcar un hueco, `N` para guardar y pasar a la siguiente)
y deja los `_gt.png` descargados en la misma carpeta `muestra/`. Con ellas se mide la segmentación (IoU).

### Si algo sale raro
- **Sale una sola categoría** con el nombre de la carpeta de la base: la carpeta está anidada
  (`Aplicacion base de datos/Aplicacion base de datos/camisas`). La carpeta que le pasas al comando debe
  contener **directamente** `camisas`, `chaquetas`, `pantalones`. Sube el contenido un nivel, borra
  `datos/` y vuelve a correr `inventario`.
- **Muchos `casi_duplicado`:** mira esas fotos con el filtro "Con avisos" de la hoja de revisión. Dos fotos son
  casi duplicadas solo si tienen la misma silueta, el mismo color y la misma proporción (`DatasetConfig`).
- **`dataset.csv` abierto en Excel** bloquea el archivo en Windows: ciérralo antes de volver a correr.

### Reglas de la división
- Entrenamiento (70 %) sirve para ajustar parámetros. **Prueba (30 %) se usa una sola vez, al final.**
- Las fotos casi iguales (la misma foto reescalada o recomprimida) siempre caen en el mismo conjunto.
- Volver a correr `inventario` conserva ids, división y muestra. `--redividir` los rehace: úsalo solo
  antes de marcar máscaras; después, mover fotos entre conjuntos invalida la medición final.
- Los originales nunca se tocan ni se renombran.

## Paso 1: preprocesamiento

```
python -m closet preprocesar "Aplicacion base de datos" --salida resultados/preprocesado
python -m closet preprocesar "Aplicacion base de datos/pantalones/foto.jpg" --salida resultados/prueba --comparar
```

Genera por cada foto `<nombre>_norm.png` (y con `--comparar`, `<nombre>_comparacion.png`
con la original y la normalizada lado a lado) y un `calidad.csv` con las métricas, la categoría
(nombre de la carpeta) y las advertencias (`borrosa`, `oscura`, `sobreexpuesta`, `bajo_contraste`,
`baja_resolucion`).

Opciones: `--lado-max 800`, `--sigma 1.0`, `--sin-balance`, `--sin-contraste`.

## Paso 2: segmentación

```
python -m closet segmentar "ruta/foto.jpg" --salida resultados/seg       # una foto o una carpeta
python -m closet evaluar --muestra muestra                                # IoU sobre las máscaras de ENTRENAMIENTO
```
`segmentar` guarda por foto `_mascara.png`, `_sin_fondo.png` (RGBA) y `_clusters.png` (qué agrupó K-means).
`evaluar` imprime IoU / precisión / recall por foto y por categoría, marca con `<-- revisar` las que quedan bajo
0.8 y guarda en `resultados/evaluacion/` una imagen por foto (verde = acierto, rojo = faltó, azul = sobró) y
`evaluacion.csv`. Las máscaras de **prueba** solo se miden con `--conjunto prueba --prueba-final`, una vez, al final.

Parámetros (`SegmentacionConfig`, también como opciones: `--k --textura --suavizado --nucleo --cierre --ventana
--umbral-centro --sin-preproceso`). Para ajustar, cambia **uno** a la vez y compara el IoU total de entrenamiento.

**Resultado actual (24 máscaras de entrenamiento):** IoU medio 0.922 (camisas 0.943, chaquetas 0.929,
pantalones 0.895); 2 fotos bajo 0.8.

**Limitaciones conocidas**
- Fondo estampado del mismo color que la prenda (jean sobre tela floral): K-means separa por color y no puede.
- La textura hace que el borde forme su propio cluster: la máscara engorda unos 3 px.
- Prendas que casi no se distinguen del fondo, o cortadas por el marco de la foto.
- Preprocesar o no (paso 1) queda empatado en la medición (0.922 vs 0.917): la diferencia es una sola foto.

## Paso 3: normalizar la prenda

```
python -m closet normalizar "Aplicacion base de datos" --salida resultados/normalizacion
```
`segmentar` y `normalizar` omiten las fotos que excluiste en `datos/dataset.csv` (usa `--todas` para procesarlas igual).
Por cada foto guarda `<nombre>_norm.png` (la prenda en un lienzo de 256 x 256, fondo transparente), y en la carpeta
de salida `normalizacion.csv` (alertas, contraste, giro, proporción alto/ancho...) y `hoja.png`, una hoja con todas
las prendas lado a lado para revisarlas de un vistazo (**borde rojo = con alerta**).

Qué hace: calcula el eje principal de la máscara; si la prenda es claramente alargada (pantalones) e inclinada
menos de 45°, la endereza; recorta la caja de la prenda y la escala (sin deformarla) para que ocupe el lienzo con un
margen fijo. El color se toma de la foto original, no de la corregida. La proporción alto/ancho se guarda en
`ratio_aspecto`, porque en un lienzo cuadrado ya no se ve.

No hace (a propósito): girar más de 45° (un eje horizontal es ambiguo en camisas con mangas abiertas) ni corregir una
prenda boca abajo.

**Alertas** (`AlertasConfig`, orientativas; calibradas con solo 24 fotos): `prenda_muy_grande`, `prenda_muy_chica`,
`prenda_parecida_fondo` (poco contraste con el fondo) y `fondo_complejo` (piso, cama o tela estampada). En entrenamiento
saltan justo en las dos segmentaciones malas. **No atrapan todo**: un pedazo de piso pegado a una prenda de color muy
distinto, por ejemplo, pasa sin alerta.

## Pruebas

```
python -m pytest
```

## Reglas del diseño

- La imagen **normalizada** es para segmentar. El **color real** de la prenda se mide sobre
  `ResultadoPreproceso.original` (sin corregir) usando la máscara obtenida.
- Las advertencias de calidad son orientativas: nunca descartan una foto por sí solas, y sus
  umbrales (`CalidadConfig`) hay que calibrarlos con fotos reales.
- El balance de blancos de mundo gris falla cuando un fondo o una prenda de un solo color domina la
  foto; por eso sus ganancias están limitadas (`ganancia_max`).
- Versión 1: una sola prenda sobre un fondo, sin personas. Categorías: camisas, chaquetas, pantalones.
