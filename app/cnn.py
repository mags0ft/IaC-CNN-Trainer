"""
Diese Datei beschäftigt sich mit der Anzeige und Verwaltung von trainierten CNNs auf Basis von
bestehenden Architekturen.

Der Workflow ist also wie uns schon bekannt, nur jetzt mit schönem UI:

    +-------------------+       +-------------------+       +------------+
    |   Architekturen   | ----> |  trainierte CNNs  | ----> |  Payloads  |
    +-------------------+       +-------------------+       +------------+

"""

import os
from secrets import token_hex
from flask import Blueprint, flash, redirect, render_template, request, url_for
from app.api import train_cnn
from app.util import generate_epic_name, get_cnns


cnn_bp = Blueprint(
    "cnn", __name__, template_folder="templates", static_folder="static", url_prefix="/cnn"
)


@cnn_bp.route("/list")
def view_all():
    """
    Zeigt alle verfügbaren CNNs an.
    """

    return render_template("cnn/list_cnns.html", cnns=get_cnns())


@cnn_bp.route("/train/<arch_name>")
def train(arch_name: str):
    """
    Zeigt die Seite zum Trainieren einer neuen CNN-Architektur an.
    """

    name = f"{arch_name}_{generate_epic_name().replace(' ', '_')}_{token_hex(3)}"

    return render_template("cnn/start_training.html", arch_name=arch_name, run_name=name)


@cnn_bp.route("/run-training", methods=["POST"])
def start_training():
    """
    Startet das Training einer CNN-Architektur. Leitet auf die Seite zum Anzeigen des Trainings
    weiter.
    """

    arch_name = request.form.get("arch_name", "").strip()
    run_name = request.form.get("run_name", "").strip()

    file_path = os.path.join(".", "data", "archs", f"{arch_name}.json")
    train_cnn(file_path, run_name)

    return redirect(url_for("cnn.view_all"))


@cnn_bp.route("/delete/<run_name>")
def delete(run_name: str):
    """
    Löscht ein trainiertes CNN.
    """

    cnn_path = os.path.join(".", "data", "cnns", run_name)

    if not os.path.exists(cnn_path):
        flash(f"Das CNN '{run_name}' existiert nicht!", "danger")
        return redirect(url_for("cnn.view_all"))

    for root, dirs, files in os.walk(cnn_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

    os.rmdir(cnn_path)

    return redirect(url_for("cnn.view_all"))


@cnn_bp.route("/view/<run_name>")
def view(run_name: str):
    """
    Zeigt die Details eines trainierten CNNs an.
    """

    cnn_path = os.path.join(".", "data", "cnns", run_name, "model.json")

    if not os.path.exists(cnn_path):
        return render_template("cnn/error.html", message="CNN not found.")

    with open(cnn_path, "r") as f:
        cnn_data = f.read()

    return render_template("cnn/view_cnn.html", run_name=run_name, cnn_data=cnn_data)
