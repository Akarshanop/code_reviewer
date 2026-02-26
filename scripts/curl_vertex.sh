#!/usr/bin/env bash
# Test Vertex AI (Gemini) with service account in backend/service.json.
# Run from backend/: ./scripts/curl_vertex.sh
# Or: cd backend && bash scripts/curl_vertex.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BACKEND_DIR"

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-development-484811}"
REGION="${VERTEX_REGION:-us-central1}"
MODEL="${VERTEX_MODEL:-gemini-1.5-flash}"

echo "Getting access token from service.json..." >&2
TOKEN=$(poetry run python scripts/get_vertex_token.py)

URL="https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/publishers/google/models/${MODEL}:generateContent"

echo "Calling Vertex AI generateContent..." >&2
curl -sS -X POST "$URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"role": "user", "parts": [{"text": "Say hello in one sentence."}]}],
    "generationConfig": {"maxOutputTokens": 128, "temperature": 0.2}
  }'

echo ""
