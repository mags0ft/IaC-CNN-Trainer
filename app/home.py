"""
Diese Datei definiert die Home-Seite der Anwendung.
"""

from flask import Blueprint, render_template


home_bp = Blueprint("home", __name__, template_folder="templates")


@home_bp.route("/")
def home():
    """
    Hier wird einfach nur die Home-Seite angezeigt. Mehr nicht.
    """

    return render_template("home.html")
