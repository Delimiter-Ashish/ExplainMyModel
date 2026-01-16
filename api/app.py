from flask import Flask
from api.routes import api_blueprint
from api.dashboard import dashboard_blueprint
import yaml

def create_app():
    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = "uploads"

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    app.config["APP_CONFIG"] = config
    app.register_blueprint(api_blueprint)

    return app

app = create_app()
app.register_blueprint(dashboard_blueprint)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )

