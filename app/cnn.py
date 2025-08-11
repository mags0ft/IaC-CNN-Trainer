"""
Diese Datei beschäftigt sich mit der Anzeige und Verwaltung von trainierten CNNs auf Basis von
bestehenden Architekturen.

Der Workflow ist also wie uns schon bekannt, nur jetzt mit schönem UI:

    +-------------------+       +-------------------+       +------------+
    |   Architekturen   | ----> |  trainierte CNNs  | ----> |  Payloads  |
    +-------------------+       +-------------------+       +------------+

"""

from flask import Blueprint, render_template


cnn_bp = Blueprint(
    "cnn", __name__, template_folder="templates", static_folder="static", url_prefix="/cnn"
)


@cnn_bp.route("/list")
def view_all():
    """
    Zeigt alle verfügbaren CNNs an.
    """

    return render_template("cnn/list_cnns.html")
