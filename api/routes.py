import os
import uuid
import pandas as pd
import yaml

from flask import Blueprint, request, render_template, jsonify

from engine.pipeline import run_pipeline

api_blueprint = Blueprint("api", __name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@api_blueprint.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@api_blueprint.route("/analyze", methods=["POST"])
def analyze():
    model_type = request.form.get("model", "xgboost")
    target = request.form.get("target")

    if not target:
        return jsonify({"error": "Target column is required"}), 400

    # Handle CSV upload or URL
    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)
        csv_source = filepath

    elif request.form.get("csv_url"):
        csv_source = request.form.get("csv_url")

    else:
        return jsonify({"error": "Provide a CSV file or CSV URL"}), 400

    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_config = config["models"].get(model_type, {})

    payload = run_pipeline(
        csv_source=csv_source,
        target=target,
        model_type=model_type,
        model_config=model_config,
        output_dir="artifacts"
    )

    return render_template("result.html", payload=payload)

