# Closet Digital

Segmenta y clasifica fotos de prendas usando **procesamiento de imágenes puro**: NumPy para todo el
procesamiento y Pillow solo para abrir y guardar archivos. K-means, los filtros, la morfología y
(más adelante) las características de Haralick están escritos a mano. Sin deep learning.

## Pipeline

| Paso | Módulo | Qué hace | Estado |
|---|---|---|---|
| 0 | `dataset`, `revision` | Inventariar la base, descartar fotos, dividir entrenamiento / prueba, elegir la muestra de máscaras | ✅ listo |
| 1 | `imagen_io`, `preprocesamiento`, `calidad` | Leer, unificar y revisar la foto | ✅ listo |
| 2 | `segmentacion` | K-means + regiones conectadas -> máscara de la prenda | por migrar desde `segmentar_camisa_kmeans.py` |
| 3 | `normalizacion` | Recortar, alinear y escalar la prenda | pendiente |
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
│   └── cli.py               comandos: python -m closet ...
├── herramientas/
│   └── anotador_mascaras.html   dibujar a mano la máscara de las fotos de la muestra
├── tests/                   pruebas automáticas (pytest)
├── datos/                   (se crea sola) dataset.csv, exclusiones.txt, revision/
├── Aplicacion base de datos/   tus fotos (camisas, chaquetas, pantalones)
└── segmentar_camisa_kmeans.py  script anterior (se migrará al paso 2)
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
