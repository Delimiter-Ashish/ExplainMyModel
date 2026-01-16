import json
import os

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def save_json(name: str, obj: dict):
    path = os.path.join(ARTIFACT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path

def load_json(name: str):
    path = os.path.join(ARTIFACT_DIR, f"{name}.json")
    with open(path, "r") as f:
        return json.load(f)

