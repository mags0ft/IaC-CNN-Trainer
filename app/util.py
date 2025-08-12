"""
Dieses Modul hat nur einen Auftrag.

Und er ist großartig.
"""

import os
from random import choice


EPIC_WORDS = [
    "foxtrot",
    "echo",
    "sierra",
    "charlie",
    "november",
    "tango",
    "romeo",
    "alpha",
    "delta",
    "zulu",
]


def generate_epic_name() -> str:
    """
    Generiert einen epischen Namen aus zwei zufälligen Wörtern.
    """

    return f"{choice(EPIC_WORDS)} {choice(EPIC_WORDS)}"


def get_archs() -> list[str]:
    """
    Gibt eine Liste aller verfügbaren Architekturen zurück.
    """

    return [arch[:-5] for arch in os.listdir("./data/archs") if arch.endswith(".json")]


def get_cnns() -> list[str]:
    """
    Gibt eine Liste aller verfügbaren CNNs zurück.
    """

    return [
        cnn for cnn in os.listdir("./data/cnns") if os.path.isdir(os.path.join("./data/cnns", cnn))
    ]
