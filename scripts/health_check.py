"""
Dieses Skript führt einen kurzen Check aus, ob alles mit der Installation geklappt hat.
"""

import os


def try_flask() -> bool:
    """
    Funktioniert Flask?
    """

    try:
        import flask

        _app = flask.Flask(__name__)

        return True
    except ImportError:
        return False


def try_dotenv() -> bool:
    """
    Funktioniert python-dotenv?
    """

    try:
        import dotenv

        dotenv.load_dotenv()

        return True
    except ImportError:
        return False


def main() -> None:
    """
    Führt die Checks aus und gibt das Ergebnis aus.
    """

    print("Führe Health-Check der Installation aus...")

    if not try_flask():
        print("Flask funktioniert nicht. Installationsskript ausgeführt?")
        return

    if not try_dotenv():
        print("python-dotenv funktioniert nicht.")
        return

    if not os.path.isfile("./.venv-training/bin/python"):
        print("Die venv für das Training fehlt anscheinend!")
        return

    print("Alles in Ordnung! Es kann losgehen!")


if __name__ == "__main__":
    main()
