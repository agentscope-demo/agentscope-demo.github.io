from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask, send_from_directory
from flask_cors import CORS

sys.path.insert(0, "src")


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    CORS(app)

    # serve the single-file dashboard at root
    dashboard_path = Path(__file__).parent

    @app.route("/")
    def index():
        return send_from_directory(str(dashboard_path), "dashboard.html")

    from api.routes import bp
    app.register_blueprint(bp)

    logging.getLogger(__name__).info("Flask app created")
    return app


if __name__ == "__main__":
    from tom.logger import setup_logging
    setup_logging("INFO")
    app = create_app()
    app.run(debug=True, port=5000, host="0.0.0.0")