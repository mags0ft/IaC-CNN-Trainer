"""
Hauptdatei für unsere Web-Oberfläche. Hier werden die Blueprints registriert und die Anwendung
gestartet.
"""

from flask import Flask
import os
import json


CONFIG_PATH = "./config.json"


def load_config() -> dict:
    """
    Lädt die Konfiguration für das UI und die Schnittstelle aus der Datei.
    """

    with open(CONFIG_PATH, "r") as file:
        return json.load(file)


def ensure_data_directory_exists():
    """
    Stellt sicher, dass das Datenverzeichnis existiert. Wenn nicht, wird es erstellt.
    """

    if not os.path.isdir("./data"):
        os.mkdir("./data")


def main() -> None:
    """
    Hier wird die App initialisiert.
    """

    ensure_data_directory_exists()

    app = Flask(__name__)
    app.config["TRAINER_CONF"] = load_config()

    # Starten der Anwendung
    app.run(port=8080)


if __name__ == "__main__":
    main()
