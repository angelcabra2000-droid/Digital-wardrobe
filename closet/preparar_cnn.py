"""Instala pesos públicos; el análisis de fotos nunca realiza descargas.

python -m closet.preparar_cnn --entrenar
"""
import argparse
import hashlib
from pathlib import Path
import urllib.request
from .red_neuronal import MODELS

URL='https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_cloth_seg.onnx'
MD5='2434d1f3cb744e0e49386c906e5a08bb'


def prepare(filename='u2net_cloth_seg.onnx', checksum=MD5):
    MODELS.mkdir(parents=True,exist_ok=True)
    destination = MODELS / filename
    if destination.is_file() and hashlib.md5(destination.read_bytes()).hexdigest()==checksum:
        return destination
    temporary=destination.with_suffix('.download')
    try:
        with urllib.request.urlopen(URL.rsplit('/', 1)[0] + '/' + filename, timeout=120) as response,temporary.open('wb') as stream:
            while block:=response.read(1024*1024):stream.write(block)
        if hashlib.md5(temporary.read_bytes()).hexdigest()!=checksum:
            raise ValueError('Los pesos no coinciden con la suma oficial. No se instalaron.')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--entrenar', action='store_true')
    parser.add_argument('--estampados', action='store_true')
    parser.add_argument('--solo-segmentacion', action='store_true')
    args = parser.parse_args()
    print(prepare())
    if not args.solo_segmentacion:
        print(prepare('u2netp.onnx', '8e83ca70e441ab06c318d82300c84806'))
    if args.estampados:
        from huggingface_hub import snapshot_download

        snapshot_download(
            'openai/clip-vit-base-patch32',
            revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
            local_dir=MODELS / 'clip',
            allow_patterns=['*.json', 'merges.txt', 'pytorch_model.bin'],
        )
    if args.entrenar:
        from .entrenar_cnn import train
        train()
