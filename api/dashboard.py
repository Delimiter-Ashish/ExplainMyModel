import json
from flask import Blueprint, render_template

dashboard_blueprint = Blueprint("dashboard", __name__)

@dashboard_blueprint.route("/dashboard", methods=["GET"])
def dashboard():
    try:
        with open("artifacts/payload.json", "r") as f:
            payload = json.load(f)
    except FileNotFoundError:
        payload = None

    return render_template("dashboard.html", payload=payload)
