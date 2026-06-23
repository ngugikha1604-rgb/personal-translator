from flask import Flask
from flask_cors import CORS
from routes.transcribe import transcribe_bp
from routes.analyze import analyze_bp


def create_app():
    app = Flask(__name__)
    CORS(app)

    app.register_blueprint(transcribe_bp)
    app.register_blueprint(analyze_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)
