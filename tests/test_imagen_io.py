import numpy as np
import pytest
from PIL import Image

from closet.imagen_io import aplicar_orientacion_exif, cargar_imagen, guardar_imagen, listar_imagenes

T = Image.Transpose
EQUIVALENTE_PILLOW = {
    2: T.FLIP_LEFT_RIGHT, 3: T.ROTATE_180, 4: T.FLIP_TOP_BOTTOM,
    5: T.TRANSPOSE, 6: T.ROTATE_270, 7: T.TRANSVERSE, 8: T.ROTATE_90,
}


@pytest.mark.parametrize("orientacion", range(1, 9))
def test_orientacion_coincide_con_pillow(orientacion):
    img = np.random.default_rng(0).integers(0, 256, (4, 7, 3), dtype=np.uint8)
    if orientacion in EQUIVALENTE_PILLOW:
        esperado = np.asarray(Image.fromarray(img).transpose(EQUIVALENTE_PILLOW[orientacion]))
    else:
        esperado = img
    assert np.array_equal(aplicar_orientacion_exif(img, orientacion), esperado)


def test_cargar_respeta_la_orientacion_exif(tmp_path):
    img = np.random.default_rng(1).integers(0, 256, (4, 7, 3), dtype=np.uint8)
    exif = Image.Exif()
    exif[274] = 6  # girar 90° en sentido horario
    Image.fromarray(img).save(tmp_path / "foto.png", exif=exif)

    cargada = cargar_imagen(tmp_path / "foto.png")

    assert cargada.shape == (7, 4, 3)
    assert np.array_equal(cargada, np.rot90(img, k=-1))


def test_cargar_devuelve_rgb_uint8_contiguo(tmp_path):
    gris = np.full((5, 6), 120, dtype=np.uint8)
    Image.fromarray(gris).save(tmp_path / "gris.png")
    img = cargar_imagen(tmp_path / "gris.png")
    assert img.shape == (5, 6, 3) and img.dtype == np.uint8 and img.flags["C_CONTIGUOUS"]


def test_transparencia_se_compone_sobre_blanco(tmp_path):
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)  # todo transparente
    Image.fromarray(rgba, "RGBA").save(tmp_path / "t.png")
    assert (cargar_imagen(tmp_path / "t.png") == 255).all()


def test_guardar_crea_carpetas(tmp_path):
    destino = tmp_path / "a" / "b" / "x.png"
    guardar_imagen(np.zeros((3, 3, 3), dtype=np.uint8), destino)
    assert destino.exists()


def test_listar_imagenes_recorre_subcarpetas_y_filtra(tmp_path):
    (tmp_path / "camisas").mkdir()
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "camisas" / "b.PNG").write_bytes(b"x")
    (tmp_path / "notas.txt").write_text("hola")
    nombres = [p.name for p in listar_imagenes(tmp_path)]
    assert sorted(nombres) == ["a.jpg", "b.PNG"]
