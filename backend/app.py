import os
from flask import Flask, send_from_directory
from flask_cors import CORS
from routes.transcribe import transcribe_bp
from routes.analyze import analyze_bp


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


def create_app():
    app = Flask(__name__)
    CORS(app)

    app.register_blueprint(transcribe_bp)
    app.register_blueprint(analyze_bp)

    @app.route("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.route("/<path:path>")
    def static_files(path):
        return send_from_directory(FRONTEND_DIR, path)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)
