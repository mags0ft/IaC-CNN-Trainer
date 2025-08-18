"""
Diese Datei beschäftigt sich mit der Anzeige und Verwaltung von trainierten CNNs auf Basis von
bestehenden Architekturen.

Der Workflow ist also wie uns schon bekannt, nur jetzt mit schönem UI:

    +-------------------+       +-------------------+       +------------+
    |   Architekturen   | ----> |  trainierte CNNs  | ----> |  Payloads  |
    +-------------------+       +-------------------+       +------------+

"""

import json
import os
from secrets import token_hex
from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from app.api import empty_scratchpad_folder, train_cnn, generate_uart_payloads, convert_architecture
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

    info_file_path = os.path.join(cnn_path, "info.json")
    if not os.path.exists(info_file_path):
        flash(f'Das CNN "{run_name}" ist noch nicht fertig trainiert!', "danger")
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

    cnn_path = os.path.join(".", "data", "cnns", run_name)
    info_file_path = os.path.join(cnn_path, "info.json")

    if not os.path.exists(info_file_path):
        flash(f'Das CNN "{run_name}" ist noch nicht fertig trainiert!', "danger")
        return redirect(url_for("cnn.view_all"))

    with open(info_file_path, "r") as f:
        run_info = json.load(f)

    return render_template(
        "cnn/view_cnn.html",
        run_name=run_name,
        run_info=run_info,
        original_cnn=current_app.config["TRAINER_CONF"]["original_cnn"],
    )


@cnn_bp.route("/get-model-graph/<run_name>/<filename>")
def get_model_graph(run_name: str, filename: str):
    """
    Liefert das Bild eines Modell-Diagramms zurück.
    """

    cnn_path = os.path.join(".", "data", "cnns", run_name)

    if not os.path.exists(cnn_path):
        flash(f"Das CNN '{run_name}' existiert nicht!", "danger")
        return redirect(url_for("cnn.view_all"))

    return send_from_directory(cnn_path, filename + ".png")


@cnn_bp.route("/gen-payloads/<run_name>")
def gen_payloads(run_name: str):
    """
    Generiert die Payloads für einen bestehenden CNN-Run.
    """

    return render_template("cnn/gen_payloads.html", run_name=run_name)


@cnn_bp.route("/start-uart-payload-generation", methods=["POST"])
def start_uart_payload_generation():
    """
    Startet die Generierung der UART-Payloads.
    """

    def _count_layers(layers: list) -> int:
        return sum(1 for layer in layers if layer["type"] in ["Conv2D", "Dense"])

    run_name = request.form.get("run_name", "").strip()
    run_path = os.path.join(".", "data", "cnns", run_name, "best_model.h5")
    info_path = os.path.join(".", "data", "cnns", run_name, "info.json")

    with open(info_path, "r") as f:
        cnn_info = json.load(f)

    arch = cnn_info["arch"]
    arch_path = os.path.join(".", "data", "archs", f"{arch}.json")

    with open(arch_path, "r") as f:
        arch_info = json.load(f)

    empty_scratchpad_folder()
    results = convert_architecture(run_path)
    if results["exit_code"] == 0:
        generate_uart_payloads(str(_count_layers(arch_info["layers"])))

    print(arch_info["layers"])

    return render_template(
        "cnn/payload_feedback.html",
        payload_logs=results["stdout"] + "\n\n" + results["stderr"],
        run_name=run_name,
        num_layers=_count_layers(arch_info["layers"]),
        exit_code=results["exit_code"],
    )
