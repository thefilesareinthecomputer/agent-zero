"""Start the Agent Zero API server.

Usage: python -m bridge.api_run
"""

import sys

import uvicorn

from agent.config import API_PORT, API_TOKEN


def main():
    if not API_TOKEN:
        print("Error: API_TOKEN not set in .env")
        sys.exit(1)
    if len(API_TOKEN) < 32:
        print("Error: API_TOKEN must be at least 32 characters")
        sys.exit(1)

    print(f"Starting Agent Zero API on 127.0.0.1:{API_PORT}")
    uvicorn.run(
        "bridge.api:app",
        host="127.0.0.1",
        port=API_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
