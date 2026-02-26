#!/usr/bin/env python3
"""
Print a Google Cloud access token for the service account in service.json.
Use: TOKEN=$(python scripts/get_vertex_token.py) && curl -H "Authorization: Bearer $TOKEN" ...
"""
import sys
from pathlib import Path

# Resolve service.json path (run from backend or project root)
backend_dir = Path(__file__).resolve().parent.parent
service_file = backend_dir / "service.json"
if not service_file.exists():
    service_file = Path("service.json")
if not service_file.exists():
    print("ERROR: service.json not found", file=sys.stderr)
    sys.exit(1)

def main():
    try:
        from google.oauth2 import service_account
    except ImportError:
        print("ERROR: Install google-auth: pip install google-auth", file=sys.stderr)
        sys.exit(1)

    creds = service_account.Credentials.from_service_account_file(
        str(service_file),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    import google.auth.transport.requests
    request = google.auth.transport.requests.Request()
    creds.refresh(request)
    print(creds.token)

if __name__ == "__main__":
    main()
