
import subprocess
import threading
import time
import requests
from flask import Flask
from pyngrok import ngrok

def setup_ngrok():
    """Starts a public ngrok tunnel."""
    # Ensure ngrok is closed before starting a new one
    try:
        ngrok.kill()
    except:
        pass # Ignore errors if ngrok is not running

    try:
        # Start a public tunnel on port 5000
        # NOTE: If you are prompted for an ngrok auth token, you need to
        # register for a free account at ngrok.com and run:
        # !ngrok config add-authtoken YOUR_AUTH_TOKEN
        tunnel = ngrok.connect(5000)
        public_url = tunnel.public_url
        return public_url
    except Exception as e:
        print(f"Failed to start ngrok tunnel: {e}")
        return None

def run_with_ngrok(app: Flask):
    """Starts the Flask app and an ngrok tunnel."""

    def run_app():
        # Running with use_reloader=False and debug=False is crucial for threading
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

    # Start the Flask app in a separate thread
    flask_thread = threading.Thread(target=run_app)
    flask_thread.daemon = True
    flask_thread.start()

    # Set up and get the public URL from ngrok
    time.sleep(2) # Wait a moment for the Flask server to start binding
    public_url = setup_ngrok()

    if public_url:
        print("\n" + "="*70)
        print(f"🌐 Flask App is LIVE! Public API Endpoint: {public_url}/predict")
        print(f"🌐 Health Check: {public_url}/health")
        print("💡 Use this public HTTPS URL to send your POST requests (e.g., from Postman).")
        print("="*70 + "\n")
    else:
        print("🚨 Could not establish public tunnel. Flask app is running internally.")
