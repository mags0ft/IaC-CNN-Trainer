"""
Diese Datei definiert alle Routen, die zur Anzeige, Bearbeitung und Verwaltung
von CNN-Architekturen verwendet werden.
"""

from flask import Blueprint, render_template


arch_bp = Blueprint(
    "arch", __name__, template_folder="templates", static_folder="static", url_prefix="/arch"
)


@arch_bp.route("/new")
def new():
    """
    Zeigt die Seite zum Erstellen einer neuen Architektur an.
    """

    return render_template("arch/new_arch.html")


@arch_bp.route("/list")
def view_all():
    """
    Zeigt alle verfügbaren Architekturen an.
    """

    return render_template("arch/list_archs.html")
