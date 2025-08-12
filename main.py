"""
Hauptdatei für unsere Web-Oberfläche. Hier werden die Blueprints registriert und die Anwendung
gestartet.
"""

import secrets
from flask import Flask
import os
import json


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


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

    for subdir in ["data", os.path.join("data", "archs"), os.path.join("data", "cnns")]:
        os.makedirs(subdir, exist_ok=True)


def main() -> None:
    """
    Hier wird die App initialisiert.
    """

    ensure_data_directory_exists()

    app = Flask(__name__)
    app.config["TRAINER_CONF"] = load_config()

    # das machen wir nur, damit flashes funktionieren, der Rest ist egal
    app.config["SECRET_KEY"] = str(secrets.token_urlsafe(32))

    app.static_folder = os.path.join("app", "static")
    app.template_folder = os.path.join("app", "templates")

    from app.home import home_bp
    from app.arch import arch_bp
    from app.cnn import cnn_bp

    app.register_blueprint(home_bp)
    app.register_blueprint(arch_bp)
    app.register_blueprint(cnn_bp)

    # Starten der Anwendung
    app.run(port=8080, debug=True)


if __name__ == "__main__":
    main()
