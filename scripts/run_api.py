"""Flask API server script.

Usage:
    python scripts/run_api.py [--host HOST] [--port PORT] [--debug]
"""

import argparse
import sys

from titanic_ml.app.routes import app
from titanic_ml.config.settings import FLASK_DEBUG, FLASK_HOST, FLASK_PORT
from titanic_ml.utils.logger import logging


def main():
    """Start Flask API server."""
    parser = argparse.ArgumentParser(description="Run Titanic ML Flask API")
    parser.add_argument(
        "--host", type=str, default=FLASK_HOST, help=f"Host address (default: {FLASK_HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=FLASK_PORT, help=f"Port number (default: {FLASK_PORT})"
    )
    parser.add_argument(
        "--debug", action="store_true", default=FLASK_DEBUG, help="Enable debug mode"
    )

    args = parser.parse_args()

    logging.info("=" * 70)
    logging.info("STARTING TITANIC ML FLASK API")
    logging.info("=" * 70)
    logging.info(f"Host: {args.host}")
    logging.info(f"Port: {args.port}")
    logging.info(f"Debug: {args.debug}")
    logging.info("=" * 70)
    logging.info("\nAPI Endpoints:")
    logging.info(f"  - Home:       http://{args.host}:{args.port}/")
    logging.info(f"  - Prediction: http://{args.host}:{args.port}/prediction")
    logging.info(f"  - Health:     http://{args.host}:{args.port}/health")
    logging.info("\n" + "=" * 70)

    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except Exception as e:
        logging.error(f"Failed to start Flask app: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
