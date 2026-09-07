# Smart Closet Vision

## Estructura del proyecto

```
smart_closet_vision/
├── data/               # Pon aquí tus fotos de prueba de prendas
├── src/
│   ├── preprocessing.py   # Paso 2: preprocesamiento ✅ (listo)
│   ├── segmentation.py    # Paso 3: segmentación (siguiente)
│   ├── features.py        # Paso 4: extracción de características
│   ├── classification.py  # Paso 5: clasificación
│   ├── closet.py           # Paso 6: armario digital (almacenamiento)
│   └── pipeline.py         # Paso 7: orquestador end-to-end
├── tests/              # Pruebas de cada módulo
├── requirements.txt
└── README.md
```

## Instalación

```bash
pip install -r requirements.txt
```

## Cómo probar el preprocesamiento

1. Copia una o dos fotos tuyas de prendas dentro de la carpeta `data/`.
2. Corre:

```bash
python src/preprocessing.py data/tu_foto.jpg
```

3. Esto genera `preprocesado_resultado.jpg` en la raíz del proyecto —
   ábrelo y compáralo con la foto original. Deberías ver:
   - Mismo tamaño estandarizado (512x512 por defecto)
   - Iluminación más pareja (menos zonas quemadas o muy oscuras)
   - Ligero suavizado (menos ruido/grano)

Si el resultado se ve bien, seguimos con `segmentation.py` (Paso 3:
Otsu + Canny + morfología).
