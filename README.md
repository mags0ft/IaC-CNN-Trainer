# IaC CNN-Trainer

Dies ist ein Hilfsprogramm mit grafischer Benutzeroberfläche, das ich im Rahmen von INVENT a CHIP 2025 erstellt habe.
Es kümmert sich um das Training und die Organisation von Audio-Classification-CNNs.

## Was es kann

- das Training von CNNs anordnen, visualisieren, überwachen etc.
- die trainierten CNNs und deren Architekturen strukturiert speichern und organisieren
- Leistungsmessungen der CNNs durchführen
- Dateien für Cutecom mithilfe des uns gegebenen Skriptes erzeugen und organisiert abspeichern

Dabei ist alles hübsch aufbereitet. Ich habe großen Wert auf Codequalität gelegt. Bitte den Code der Oberfläche und des Programms im Generellen beachten.

## Installation

Bitte für die Installation des Programms folgenden Befehl ausführen:

```bash
./scripts/install.sh
```

Es handelt sich hier um eine Web-Oberfläche, die mittels Flask geschrieben wurde. Getestet in Python 3.10.
Die Installation wird sich selbst überprüfen. Um das selbst nachzuholen, sicherstellen, dass das venv `.venv-server` aktiviert ist, dann

```bash
./.venv-server/bin/python3 ./scripts/health_check.py
```

ausführen.

Das Skript wird Python 3.8 und Python 3.10 installieren, virtuelle Umgebungen erzeugen, alle Dependencies laden und dann das venv für 3.10 aktiviert lassen.

Um das GUI zu starten, bitte

```bash
./.venv-server/bin/python3 ./main.py
```

ausführen. Um aus `app/cnn.py` frei zu zitieren,

```
Der Workflow ist also wie uns schon bekannt, nur jetzt mit schönem UI:

+-----------------+     +-------------------+     +------------+
|  Architekturen  | --> |  trainierte CNNs  | --> |  Payloads  |
+-----------------+     +-------------------+     +------------+
```
