# Closet Digital

Segmenta y clasifica fotos de prendas con dos métodos: **U²-Net de ropa + CNN ResNet-18** en la web principal, y el método clásico NumPy (K-means, morfología, Haralick y KNN) en el estudio técnico. U²-Net es una variante anidada de U-Net; sus pesos de segmentación son externos preentrenados y la cabeza de clasificación ResNet-18 se adapta al catálogo local.

## Red neuronal y textura visual (actualización)

```bash
source venv/bin/activate
python -m pip install -r requirements-cnn.txt
python -m closet.preparar_cnn --entrenar --estampados
python -m closet.web --host 0.0.0.0
```

Si los pesos ya están instalados, basta iniciar `closet.web`. La inferencia funciona localmente, sin enviar las fotos a servicios de IA. En Ajustes del análisis se puede elegir CNN o clásico. La CNN utiliza 90 fotos incluidas de entrenamiento; la selección de época usó una reserva interna por grupos de 18 fotografías y obtuvo 18/18 antes del reajuste con las 90. Esa cifra selecciona el modelo, no es una prueba independiente ni una evaluación del segmentador. Las 39 imágenes reservadas de prueba no se utilizaron. Los resultados no son probabilidades calibradas.

`texturas.py` mide detalle a resolución de entrada dentro de la máscara, con contraste multiescala, patrones binarios locales y periodicidad. Distingue textura fina, marcada, repetitiva, irregular o no concluyente y separa textura de patrón. No identifica material ni tacto. Las fotos de baja resolución y las que contienen varias prendas requieren revisión. El selector superior/inferior es un recorte espacial asistido, no detección de instancias. La ficha muestra el estampado detectado por CLIP, como palmeras, flores o sin estampado. Escanear devuelve una sola prenda por foto. Puedes elegir la zona en Ajustes del análisis. Los resultados históricos deben volver a analizarse para obtener el estampado.

Los pesos grandes `datos/red_neuronal/*.onnx` y `*.pt` están excluidos de Git. `preparar_cnn` verifica los pesos de U²-Net distribuidos por [rembg](https://github.com/danielgatis/rembg); `entrenar_cnn` reconstruye el clasificador. `modelo.json` documenta los grupos, métricas y sumas. Las fotos de usuarios no se añaden al entrenamiento. Pruebas actuales: **184 satisfactorias**.

Los apartados siguientes describen principalmente el pipeline clásico original. Sus cifras históricas de IoU no son una evaluación de U²-Net ni una nueva medición de la integración.

## Pipeline

| Paso | Módulo | Qué hace | Estado |
|---|---|---|---|
| 0 | `dataset`, `revision` | Inventariar la base, descartar fotos, dividir entrenamiento / prueba, elegir la muestra de máscaras | ✅ listo |
| 1 | `imagen_io`, `preprocesamiento`, `calidad` | Leer, unificar y revisar la foto | ✅ listo |
| 2 | `segmentacion`, `evaluacion` | K-means + regiones conectadas -> máscara de la prenda; medición con IoU | ✅ listo (IoU 0.92 en entrenamiento) |
| 3 | `normalizacion`, `alertas`, `pipeline` | Recortar, enderezar y escalar la prenda; alertas de recorte dudoso | ✅ listo |
| 4 | `caracteristicas`, `atributos` | Color, forma, textura y descriptores visuales estimados | implementado |
| 5 | `clasificacion` | KNN sobre características; sugerencia revisable de categoría | implementado; validación interna 82.2% |
| Web | `web`, `frontend/gossip.html` | Interfaz, análisis, historial y clóset SQLite | implementado |

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

**Resultado histórico del método clásico (24 máscaras de entrenamiento; no reevaluado en la integración CNN):** IoU medio 0.922 (camisas 0.943, chaquetas 0.929,
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


## Estudio visual local (frontend y backend)

Desde la raíz del proyecto en macOS:

```bash
source venv/bin/activate
python -m closet.web
```

Abre http://127.0.0.1:8765 en el navegador. También puedes abrir `Abrir estudio.command` con doble clic. No requiere compilar JavaScript ni instalar Node. El entorno Python necesita las dependencias de `requirements.txt`.

El estudio permite subir imágenes de hasta 20 MB, seleccionar fotos de la base, ajustar K-means, comparar original y recorte, inspeccionar preproceso, clusters, máscara y normalización, y descargar PNG o un ZIP con todas las imágenes y las características JSON. El historial se conserva en `resultados/estudio/`. No envía imágenes a servicios externos.

El servidor `closet/web.py` usa HTTP de la biblioteca estándar y escucha únicamente en `127.0.0.1`. Ejecuta el pipeline real del proyecto en una cola de un trabajador para limitar el uso simultáneo de memoria. La interfaz está en `closet/frontend/index.html`. Para usar otro puerto: `python -m closet.web --port 8766`. Detén el servidor con Ctrl+C.

La resolución de análisis está limitada a 800 px; los PNG corresponden a esa resolución de procesamiento. Es un estudio local, no un servicio preparado para exponer en Internet.

El catálogo visual lee los 142 registros de `datos/dataset.csv`, muestra ID, categoría y conjunto, y respeta las exclusiones: las fotos excluidas se pueden consultar con el filtro, pero no se procesan desde el catálogo. Se usa `Aplicacion base de datos/` como carpeta de fotos, con `BasedeDatos/` como alternativa si la primera no existe. Los registros CSV originales no se modifican.


## Closet Gossip: interfaz del ZIP y uso desde el celular

La página principal usa las imágenes originales de `closet_gossip_azul.zip`, con sus vistas Escanear, Mi clóset y Perfil. Se adaptó el frontend a HTML/CSS/JavaScript para usar el backend existente sin agregar una segunda aplicación Gradio. La referencia Gradio original permanece en `closet/frontend/closet_gossip/app_gradio_original.txt`; su función aleatoria no se ejecuta.

```bash
source venv/bin/activate
python -m closet.web --host 0.0.0.0
```

Desde el Mac abre http://127.0.0.1:8765. Desde el celular, conectado a la misma Wi-Fi, usa `http://IP_DEL_MAC:8765`. El iniciador `Abrir Closet Gossip en celular.command` muestra esa dirección. El Mac debe permanecer encendido con el servidor activo. Si macOS pregunta por conexiones entrantes, permite las del servidor para usarlo desde otro dispositivo. No es un despliegue público en Internet.

**Escanear:** Tomar foto usa el selector de cámara nativo del navegador móvil con `capture=environment`. Elegir imagen permite cargar una foto. El cliente normaliza orientación y tamaño a un máximo de 1800 px antes de enviar JPEG al backend; el análisis usa 400, 600 u 800 px. La disponibilidad de cámara depende del navegador/dispositivo. No se ha probado físicamente en un teléfono.

**Procesamiento:** se ejecuta en módulos Python, no en los tests. `pipeline.py` preprocesa, segmenta y normaliza; `caracteristicas.py` extrae las medidas; `atributos.py` estima patrón, silueta, corte, simetría, claridad, intensidad y composición cromática; `red_neuronal.py` segmenta con U²-Net y sugiere el tipo con ResNet-18 en modo CNN; `clasificacion.py` conserva vecinos próximos para el modo clásico; `texturas.py` mide la apariencia local sobre la imagen de entrada. La pestaña técnica de resultados permite inspeccionar cada etapa.

**Clasificador clásico:** `datos/modelo_caracteristicas.json` usa únicamente 90 registros incluidos de entrenamiento. La distancia normalizada pondera forma 80%, color 8% y textura 12%. Su validación deja fuera todo el grupo de la consulta y recalcula la normalización; obtuvo 82.22% global, 77.27% en camisas, 89.74% en chaquetas y 75.86% en pantalones. Son métricas internas de entrenamiento, no una evaluación final de las 39 fotos de prueba. No hay probabilidad calibrada ni garantía para fotos nuevas. Para regenerar: `python -m closet.clasificacion --entrenar`. Si se cambia el procesamiento, se deben regenerar los JSON intermedios de `resultados/entrenamiento_caracteristicas/` antes de reentrenar.

**Mi clóset:** dos pestañas organizan las mismas prendas guardadas: Por tipo de prenda y Por color. La búsqueda también admite patrón o silueta. Se puede corregir el tipo sugerido antes de guardar. Las prendas se conservan con SQLite en `resultados/estudio/closet.sqlite3`, referenciando los análisis existentes. Guardar otra vez el mismo análisis actualiza su ficha sin duplicarla.

**Perfil:** nombre, usuario y avatar se guardan en el navegador. Es personalización local sin contraseña ni autenticación. Cada navegador tiene un identificador de clóset distinto; usar el mismo nombre en otro dispositivo no sincroniza automáticamente el perfil. La versión de red está destinada a demostraciones en una Wi-Fi de confianza. Antes de publicarla en Internet se requiere autenticación y un despliegue HTTPS.

Los patrones (liso, rayas horizontales/verticales, cuadros o irregular) son estimaciones de periodicidad del color. Silueta (alargada, intermedia, ancha), corte y simetría son descriptores geométricos de la máscara, no una clasificación de moda validada. Pliegues, sombras, orientación y errores de segmentación pueden alterar esas etiquetas. No se identifican material, talla, marca ni composición textil.
