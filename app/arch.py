"""
Diese Datei definiert alle Routen, die zur Anzeige, Bearbeitung und Verwaltung
von CNN-Architekturen verwendet werden.
"""

from flask import Blueprint, render_template


arch_bp = Blueprint("arch", __name__, template_folder="templates", url_prefix="/arch")
