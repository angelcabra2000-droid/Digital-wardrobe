"""Hoja de revisión visual de la base de imágenes.

Con 200 o más fotos conviene mirarlas todas una vez, de golpe, para descartar las que no sirven para esta
primera versión: personas con la prenda puesta, varias prendas en una foto, prendas cortadas, etc.
Esta hoja muestra una miniatura por foto, agrupada por categoría, y deja marcar las que se excluyen.

Se abre con doble clic (es un solo HTML + una carpeta de miniaturas, sin servidor). Al terminar,
el botón "Descargar exclusiones.txt" baja la lista; se guarda junto a dataset.csv y se vuelve a correr
`python -m closet inventario`, que la aplica.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dataset import MOTIVO_MANUAL, Registro
from .imagen_io import cargar_imagen, guardar_imagen
from .preprocesamiento import redimensionar_lado_max

PLANTILLA = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Revisión de la base</title>
<style>
  :root { --bg:#f5f5f2; --fg:#1d1d1b; --mute:#6b6b66; --card:#fff; --line:#d9d9d2;
          --acc:#1f6feb; --acc-fg:#fff; --warn:#b42318; --warn-bg:#fdecea; --aviso:#8a5a00; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#151513; --fg:#ece9e6; --mute:#9b9b94; --card:#222220; --line:#3a3a36;
            --acc:#58a6ff; --acc-fg:#0b0b0a; --warn:#ff7b72; --warn-bg:#3a1f1c; --aviso:#e3b341; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { position:sticky; top:0; z-index:5; background:var(--card); border-bottom:1px solid var(--line);
           padding:10px 16px; display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; }
  header strong { font-size:16px; }
  #cuenta { color:var(--mute); }
  label.filtro { cursor:pointer; }
  button { font:inherit; color:var(--acc-fg); background:var(--acc); border:0; border-radius:8px;
           padding:7px 14px; font-weight:600; cursor:pointer; }
  p.ayuda { margin:12px 16px 0; color:var(--mute); max-width:70ch; }
  h2 { margin:22px 16px 8px; font-size:17px; }
  h2 span { color:var(--mute); font-weight:400; }
  .rejilla { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px; padding:0 16px 8px; }
  .tarjeta { background:var(--card); border:2px solid var(--line); border-radius:10px; overflow:hidden;
             cursor:pointer; position:relative; display:flex; flex-direction:column; }
  .tarjeta img { width:100%; aspect-ratio:1; object-fit:contain; background:#8883; display:block; }
  .tarjeta.excluida { border-color:var(--warn); background:var(--warn-bg); }
  .tarjeta.excluida img { opacity:.35; }
  .tarjeta.excluida::after { content:"EXCLUIDA"; position:absolute; top:8px; left:8px; background:var(--warn);
           color:#fff; font-size:11px; font-weight:700; padding:2px 6px; border-radius:4px; }
  .pie { padding:6px 8px; font-size:12px; overflow-wrap:anywhere; }
  .pie b { display:block; }
  .pie em { color:var(--aviso); font-style:normal; display:block; }
  .tarjeta input { position:absolute; opacity:0; pointer-events:none; }
  .oculta { display:none; }
</style>
</head>
<body>
<header>
  <strong>Revisión de la base</strong>
  <span id="cuenta"></span>
  <label class="filtro"><input type="radio" name="f" value="todas" checked> Todas</label>
  <label class="filtro"><input type="radio" name="f" value="avisos"> Con avisos</label>
  <label class="filtro"><input type="radio" name="f" value="excluidas"> Excluidas</label>
  <button id="descargar">Descargar exclusiones.txt</button>
</header>
<p class="ayuda">Haz clic en una foto para <strong>excluirla</strong> (otra vez para volver a incluirla).
Excluye las que tengan una persona con la prenda puesta, varias prendas, la prenda cortada o que no se
vea bien. Al terminar, descarga el archivo y guárdalo junto a <code>dataset.csv</code>; luego vuelve a
correr <code>python -m closet inventario</code>.</p>
<main id="contenido"></main>
<script>
const FOTOS = __DATOS__;
const CLAVE = "closet_exclusiones___CLAVE__";
const marcadas = new Set(FOTOS.filter(f => f.excluida).map(f => f.ruta));
try {  // recordar lo marcado si se cierra la pestaña; si el navegador lo impide, no pasa nada
  const g = JSON.parse(localStorage.getItem(CLAVE) || "null");
  if (Array.isArray(g)) { marcadas.clear(); g.forEach(r => marcadas.add(r)); }
} catch (e) {}

const $ = s => document.querySelector(s);
const guardar = () => { try { localStorage.setItem(CLAVE, JSON.stringify([...marcadas])); } catch (e) {} };

function dibujar() {
  const cont = $('#contenido'); cont.innerHTML = '';
  const cats = [...new Set(FOTOS.map(f => f.categoria))].sort();
  for (const cat of cats) {
    const fotos = FOTOS.filter(f => f.categoria === cat);
    const h = document.createElement('h2'); h.id = 'h-' + cat;
    h.innerHTML = cat + ' <span></span>'; cont.appendChild(h);
    const grid = document.createElement('div'); grid.className = 'rejilla'; cont.appendChild(grid);
    for (const f of fotos) {
      const t = document.createElement('label'); t.className = 'tarjeta'; t.dataset.ruta = f.ruta;
      t.dataset.avisos = f.avisos ? '1' : '';
      t.innerHTML = '<input type="checkbox"><img loading="lazy" alt=""><div class="pie"><b></b><em></em></div>';
      t.querySelector('img').src = f.miniatura;
      t.querySelector('b').textContent = f.id + ' · ' + f.ruta.split('/').pop();
      t.querySelector('em').textContent = f.avisos.split(';').filter(Boolean).join(' · ');
      t.querySelector('input').addEventListener('change', () => {
        marcadas.has(f.ruta) ? marcadas.delete(f.ruta) : marcadas.add(f.ruta);
        guardar(); actualizar();
      });
      grid.appendChild(t);
    }
  }
  actualizar();
}

function actualizar() {
  const filtro = document.querySelector('input[name=f]:checked').value;
  document.querySelectorAll('.tarjeta').forEach(t => {
    const ex = marcadas.has(t.dataset.ruta);
    t.classList.toggle('excluida', ex);
    t.querySelector('input').checked = ex;
    const visible = filtro === 'todas' || (filtro === 'avisos' && t.dataset.avisos) || (filtro === 'excluidas' && ex);
    t.classList.toggle('oculta', !visible);
  });
  document.querySelectorAll('h2').forEach(h => {
    const cat = h.id.slice(2), fotos = FOTOS.filter(f => f.categoria === cat);
    h.querySelector('span').textContent = '(' + fotos.length + ' fotos, ' +
      fotos.filter(f => marcadas.has(f.ruta)).length + ' excluidas)';
  });
  $('#cuenta').textContent = FOTOS.length + ' fotos · ' + marcadas.size + ' excluidas';
}

document.querySelectorAll('input[name=f]').forEach(r => r.addEventListener('change', actualizar));
$('#descargar').addEventListener('click', () => {
  const texto = '# Fotos excluidas de la base (una ruta por linea)\\n' + [...marcadas].sort().join('\\n') + '\\n';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([texto], { type: 'text/plain;charset=utf-8' }));
  a.download = 'exclusiones.txt'; a.click();
});
dibujar();
</script>
</body>
</html>
"""


def generar_revision(registros: list[Registro], raiz: str | Path, salida: str | Path, lado: int = 240) -> Path:
    """Crea `salida/revision.html` y `salida/miniaturas/`. Devuelve la ruta del HTML.

    Se muestran las fotos incluidas y las excluidas a mano (para poder volver a incluirlas).
    """
    raiz, salida = Path(raiz), Path(salida)
    carpeta_min = salida / "miniaturas"
    carpeta_min.mkdir(parents=True, exist_ok=True)

    datos = []
    for r in sorted(registros, key=lambda r: (r.categoria, r.ruta)):
        if not (r.incluir or r.motivo == MOTIVO_MANUAL) or not r.sha1:
            continue
        miniatura = carpeta_min / f"{r.id}.jpg"
        if not miniatura.exists():
            guardar_imagen(redimensionar_lado_max(cargar_imagen(raiz / r.ruta), lado), miniatura)
        datos.append({
            "id": r.id, "ruta": r.ruta, "categoria": r.categoria, "avisos": r.avisos,
            "excluida": not r.incluir, "miniatura": f"miniaturas/{r.id}.jpg",
        })

    # La clave del recuerdo del navegador incluye el estado inicial: si el inventario cambia
    # (por ejemplo, editaste dataset.csv), las marcas viejas del navegador dejan de aplicarse.
    inicial = "|".join(sorted(d["ruta"] for d in datos if d["excluida"]))
    clave = ("".join(c for c in str(raiz.resolve()) if c.isalnum())[-30:]
             + hashlib.sha1(inicial.encode("utf-8")).hexdigest()[:8])
    html = (PLANTILLA
            .replace("__DATOS__", json.dumps(datos, ensure_ascii=False).replace("</", "<\\/"))
            .replace("__CLAVE__", clave))
    destino = salida / "revision.html"
    destino.write_text(html, encoding="utf-8")
    return destino
