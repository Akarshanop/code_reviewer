from supabase import create_client, Client
from fastapi import Depends
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

async def get_supabase_client() -> Client:
    client: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    try:
        yield client
    finally:
        pass
    
async def test_supabase_connection(client=Depends(get_supabase_client)):
    try:
        response = client.table('audits').select('*', count='exact').limit(1).execute()
        logger.info(f"Supabase connection successful. Table count: {response.count}")
        return {"status": "connected", "count": response.count}
    except Exception as e:
        logger.error(f"Supabase test failed: {e}")
        raise