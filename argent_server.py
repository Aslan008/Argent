"""Entry point for the Argent backend (for the Tauri/web GUI).

The terminal version stays main.py — this is the second client's server, on the
same core. Run: python argent_server.py  (listens on 127.0.0.1:8756).
"""

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8756


def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    import uvicorn
    from src.server.app import app
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
