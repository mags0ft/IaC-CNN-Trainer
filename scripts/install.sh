#!/bin/bash

# als Erstes möchten wir sicherstellen, ob pyenv installiert ist
if ! command -v pyenv &> /dev/null; then
    echo "pyenv ist nicht installiert. Bitte nachholen!"
    exit 1
fi

# zuerst die Dependencies fürs Training installieren
pyenv install 3.8.10 && pyenv global 3.8.10 && python3 -m venv .venv-training
source .venv-training/bin/activate
pip install --no-deps -r ./requirements-training.txt
deactivate

# jetzt kommen die für unser GUI
pyenv install 3.10.12 && pyenv global 3.10.12 && python3 -m venv .venv-server
source .venv-server/bin/activate
pip install -r ./requirements-server.txt

# fertig! Mal schauen, ob etwas schief gegangen ist :)
echo "Alles installiert! Wir schauen jetzt automatisch, ob alles geklappt hat."
./.venv-server/bin/python3 ./scripts/health_check.py
