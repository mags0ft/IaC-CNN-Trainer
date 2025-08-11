"""
Diese Datei beschäftigt sich mit der Anzeige und Verwaltung von trainierten CNNs auf Basis von
bestehenden Architekturen.

Der Workflow ist also wie uns schon bekannt, nur jetzt mit schönem UI:

    +-------------------+       +-------------------+       +------------+
    |   Architekturen   | ----> |  trainierte CNNs  | ----> |  Payloads  |
    +-------------------+       +-------------------+       +------------+

"""

from flask import Blueprint, render_template


arch_bp = Blueprint("arch", __name__, template_folder="templates", url_prefix="/arch")
