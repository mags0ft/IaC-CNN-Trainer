"""
Diese Datei definiert alle Routen, die zur Anzeige, Bearbeitung und Verwaltung
von CNN-Architekturen verwendet werden.
"""

import json
import string
from flask import Blueprint, flash, redirect, render_template, request, url_for
import os

from app.util import get_archs


arch_bp = Blueprint(
    "arch", __name__, template_folder="templates", static_folder="static", url_prefix="/arch"
)


@arch_bp.route("/new")
def new():
    """
    Zeigt die Seite zum Erstellen einer neuen Architektur an.
    """

    return render_template("arch/new_arch.html")


@arch_bp.route("/new/create", methods=["POST"])
def create_arch():
    """
    Verarbeitet das Formular zum Erstellen einer neuen Architektur.
    """

    arch_name = request.form.get("name", "").strip().lower().replace(" ", "_")

    if arch_name == "":
        flash("Der Name darf nicht leer sein!", "danger")
        return redirect(url_for("arch.new"))

    for char in request.form["name"]:
        if char not in string.ascii_lowercase + string.digits + "_":
            flash(
                "Der Name darf nur ASCII-Buchstaben, Zahlen und Unterstriche enthalten!",
                "danger",
            )
            return redirect(url_for("arch.new"))

    arch_path = os.path.join(".", "data", "archs", f"{arch_name}.json")

    with open(arch_path, "w") as f:
        json.dump(
            {
                "name": arch_name,
                "input_shape": (16000, 1, 1),
                "layers": [],
                "training_lr": 0.0003,
                "training_epochs": 16,
                "training_batch_size": 64,
            },
            f,
            indent=4,
        )

    flash(f'Die Architektur "{arch_name}" wurde erfolgreich erstellt!', "success")
    return redirect(url_for("arch.view_all"))


@arch_bp.route("/list")
def view_all():
    """
    Zeigt alle verfügbaren Architekturen an.
    """

    return render_template("arch/list_archs.html", archs=get_archs())


@arch_bp.route("/edit/<arch_name>")
def edit(arch_name: str):
    """
    Zeigt die Seite zum Bearbeiten einer Architektur an.
    """

    arch_path = os.path.join(".", "data", "archs", f"{arch_name}.json")

    if not os.path.isfile(arch_path):
        flash(f"Die Architektur '{arch_name}' existiert nicht!", "danger")
        return redirect(url_for("arch.view_all"))

    with open(arch_path, "r") as f:
        arch_data = json.load(f)

    return render_template(
        "arch/edit_arch.html",
        arch_name=arch_name,
        layers="\n".join([json.dumps(i) for i in arch_data["layers"]]),
        training_lr=arch_data["training_lr"],
        training_epochs=arch_data["training_epochs"],
        training_batch_size=arch_data["training_batch_size"],
    )


@arch_bp.route("/edit/<arch_name>/save", methods=["POST"])
def save(arch_name: str):
    """
    Speichert die Änderungen an einer Architektur.
    """

    arch_path = os.path.join(".", "data", "archs", f"{arch_name}.json")

    if not os.path.isfile(arch_path):
        flash(f"Die Architektur '{arch_name}' existiert nicht!", "danger")
        return redirect(url_for("arch.view_all"))

    with open(arch_path, "r") as f:
        arch_data = json.load(f)

    arch_data["layers"] = [
        json.loads(i) for i in request.form.get("arch-json", "").strip().splitlines()
    ]

    arch_data["training_lr"] = float(request.form.get("learning-rate", arch_data["training_lr"]))
    arch_data["training_epochs"] = int(request.form.get("epochs", arch_data["training_epochs"]))
    arch_data["training_batch_size"] = int(
        request.form.get("batch-size", arch_data["training_batch_size"])
    )

    with open(arch_path, "w") as f:
        json.dump(arch_data, f, indent=4)

    flash(f'Die Architektur "{arch_name}" wurde erfolgreich aktualisiert!', "success")
    return redirect(url_for("arch.view_all"))


@arch_bp.route("/edit/<arch_name>/duplicate", methods=["GET", "POST"])
def duplicate(arch_name: str):
    """
    Dupliziert eine bestehende Architektur.
    """

    if request.method == "GET":
        return render_template("arch/duplicate_arch.html", arch_name=arch_name)

    arch_path = os.path.join(".", "data", "archs", f"{arch_name}.json")

    if not os.path.isfile(arch_path):
        flash(f"Die Architektur '{arch_name}' existiert nicht!", "danger")
        return redirect(url_for("arch.view_all"))

    with open(arch_path, "r") as f:
        arch_data = json.load(f)

    new_arch_name = request.form.get("new_arch_name", "").strip().lower().replace(" ", "_")
    new_arch_path = os.path.join(".", "data", "archs", f"{new_arch_name}.json")

    with open(new_arch_path, "w") as f:
        json.dump(arch_data, f, indent=4)

    flash(f'Die Architektur "{arch_name}" wurde erfolgreich dupliziert!', "success")
    return redirect(url_for("arch.view_all"))


@arch_bp.route("/edit/<arch_name>/delete-prompt")
def delete_prompt(arch_name: str):
    """
    Zeigt die Bestätigungsseite zum Löschen einer Architektur an.
    """

    return render_template("arch/delete_arch.html", arch_name=arch_name)


@arch_bp.route("/edit/<arch_name>/delete")
def delete(arch_name: str):
    """
    Löscht eine Architektur.
    """

    arch_path = os.path.join(".", "data", "archs", f"{arch_name}.json")

    if not os.path.isfile(arch_path):
        flash(f"Die Architektur '{arch_name}' existiert nicht!", "danger")
        return redirect(url_for("arch.view_all"))

    os.remove(arch_path)

    flash(f"Die Architektur '{arch_name}' wurde erfolgreich gelöscht!", "success")

    return redirect(url_for("arch.view_all"))
