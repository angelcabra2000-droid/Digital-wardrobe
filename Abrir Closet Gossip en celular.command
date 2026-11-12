#!/bin/zsh
cd "$(dirname "$0")"
echo "Conecta tu celular a la misma Wi-Fi de este Mac."
closet_ip=$(/usr/sbin/ipconfig getifaddr en0)
echo "Abre en el celular: http://${closet_ip}:8765"
echo "Mantén esta terminal abierta. Ctrl+C detiene el servidor."
if /usr/bin/curl --silent --fail http://127.0.0.1:8765/ >/dev/null; then
  echo "Ya hay un servidor en ejecución. Si solo escucha en 127.0.0.1, detenlo antes de usar este iniciador."
  open "http://127.0.0.1:8765"
  exit 0
fi
(sleep 2; open "http://127.0.0.1:8765") &
exec venv/bin/python -B -m closet.web --host 0.0.0.0
