# Migration to Google Cloud Service Account Authentication

## Overview
Migrated from Gemini API key (`GEMINI_API_KEY`) to Google Cloud service account authentication for all Gemini/Vertex AI LLM calls. This provides:
- ✅ Better quota management and control
- ✅ Service account-based authentication (no API key exposed)
- ✅ Consistent with existing embedding service authentication
- ✅ Support for project-based resource isolation

## Changes Made

### 1. Configuration Files
**File:** `config/settings.py`
- Updated `get_api_key_for_model()` to return `None` for Gemini models (they use service account auth)
- Updated validation to check for `GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE` instead of `GEMINI_API_KEY`
- Gemini models now use `GOOGLE_APPLICATION_CREDENTIALS` environment variable set at startup

**File:** `.env.example`
- Removed `GEMINI_API_KEY` requirement
- Added documentation for service account setup
- Example models updated to use `gemini-2.5-flash`

### 2. Application Startup
**File:** `src/api/main.py`
- Added initialization code to set `GOOGLE_APPLICATION_CREDENTIALS` from `GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE`
- Runs at startup before any LLM calls
- Logs project ID for verification

### 3. LLM Workflow Updates
All three agent workflows now use service account authentication:

**Files Updated:**
- `src/agents/workflows/developer_flow.py` - Developer assistant queries
- `src/agents/workflows/prd_flow.py` - PRD analysis (3 LLM call sites)
- `src/agents/workflows/qa_flow.py` - QA generation

**Pattern Applied:**
```python
# Instead of:
response = await acompletion(
    model=model,
    messages=messages,
    api_key=settings.get_api_key_for_model(model),
    # ... other params
)

# Now:
llm_kwargs = {
    "model": model,
    "messages": messages,
    # ... other params
}

if "gemini" in model.lower():
    llm_kwargs["api_key"] = "dummy"  # LiteLLM uses GOOGLE_APPLICATION_CREDENTIALS
else:
    llm_kwargs["api_key"] = settings.get_api_key_for_model(model)

response = await acompletion(**llm_kwargs)
```

## Environment Setup

### Before (API Key based)
```bash
GEMINI_API_KEY=AIzaSy...
```

### After (Service Account based)
```bash
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_CLOUD_REGION=us-central1
GOOGLE_CLOUD_SERVICE_ACCOUNT_FILE=./service.json
```

### Getting Service Account Credentials
1. Go to [Google Cloud Console - Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
2. Click "Create Service Account"
3. Grant these roles:
   - `Vertex AI User` (for LLM calls)
   - `Generative AI User` (for Gemini API)
4. Create JSON key and download to your project root as `service.json`

## Benefits

| Aspect | Before (API Key) | After (Service Account) |
|--------|------------------|------------------------|
| Authentication | API Key string | Service account JSON |
| Quota Management | Hard limit on account | Per-project quota control |
| Credentials | Exposed in memory | Only in trusted `service.json` file |
| Multi-project | Not possible | Possible (one service account per project) |
| Revocation | Manual API key rotation | Simple service account key rotation |
| Compliance | Basic security | Enterprise-grade security |

## Testing

After updating your `.env` file with service account credentials:

```bash
# Test health check (includes LLM connectivity verification)
curl http://localhost:8000/api/v1/admin/health

# Test developer assistant
curl -X POST http://localhost:8000/api/v1/developer/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the main APIs?","chat_history":[]}'
```

## Backward Compatibility

The system now:
- ✅ Ignores `GEMINI_API_KEY` if set (won't error, just won't use it)
- ✅ Still supports other LLM providers (OpenAI, Anthropic, Azure) with their API keys
- ✅ Maintains all existing endpoint functionality

## Files Modified
- `config/settings.py` (2 changes)
- `src/api/main.py` (1 addition)
- `.env.example` (1 section update)
- `src/agents/workflows/developer_flow.py` (1 change)
- `src/agents/workflows/prd_flow.py` (3 changes)
- `src/agents/workflows/qa_flow.py` (1 change)

**Total:** 9 targeted updates across 6 files
