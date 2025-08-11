"""
Hier werden keine Routen definiert, sondern nur der darunterliegende Prozess für die jeweilige
Aufgabe gestartet.

z.B.: wenn wir im UI das Training einer CNN-Architektur starten, dann wird hier der Prozess
gestartet, der das Training durchführt.

oder: wenn wir eine Architektur umwandeln wollen, dann wird hier automatisch dafür gesorgt, dass das
dazugehörige Skript gestartet wird.
"""

from subprocess import Popen, PIPE
from typing import Union
from flask import current_app


def train_cnn(config_file: str, run_name: str) -> None:
    """
    Führt den Prozess für das Training einer CNN-Architektur aus. Ist non-blocking.
    """

    args = current_app.config.get("TRAINER_CONF", {})["commands"]["train_cnn"]

    Popen(args + [config_file, run_name])


def convert_architecture(run_name: str) -> dict[str, Union[str, int]]:
    """
    Wandelt in die Architektur für das FPGA um (LINA). Blocking, da das oft scheitert.

    Das zurückgegebene dict gibt `stdout`, `stderr` und den `exit_code` an:

    ```
    {
        "stdout": "...",
        "stderr": "...",
        "exit_code": 0
    }
    ```
    """

    args = current_app.config.get("TRAINER_CONF", {})["commands"]["convert_cnn"]

    process = Popen(args + [run_name], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    return {"stdout": stdout.decode(), "stderr": stderr.decode(), "exit_code": process.returncode}


def generate_uart_payloads(run_name: str) -> None:
    """
    Generiert die UART-Payloads für die FPGA-Architektur. Non-blocking.
    """

    args = current_app.config.get("TRAINER_CONF", {})["commands"]["generate_uart_payloads"]

    Popen(args + [run_name])
