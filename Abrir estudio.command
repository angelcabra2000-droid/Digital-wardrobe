#!/bin/zsh
cd "$(dirname "$0")"
if /usr/bin/curl --silent --fail http://127.0.0.1:8765/ >/dev/null; then
  open "http://127.0.0.1:8765"
  exit 0
fi
if [ ! -x venv/bin/python ]; then
  echo "Primero crea el entorno: python3 -m venv venv"
  exit 1
fi
(sleep 2; open "http://127.0.0.1:8765") &
exec venv/bin/python -m closet.web
