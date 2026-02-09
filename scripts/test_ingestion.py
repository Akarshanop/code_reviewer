#!/usr/bin/env python3
"""
Test ingestion by making HTTP requests to the backend API
"""
import asyncio
import aiohttp
import json
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.core.logger import get_logger

logger = get_logger(__name__)

API_URL = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1"


async def trigger_ingestion():
    """Trigger ingestion via API"""
    logger.info("=" * 60)
    logger.info("TRIGGERING INGESTION")
    logger.info("=" * 60)
    
    repos = settings.get_repositories()
    if not repos:
        logger.error("✗ No repositories configured")
        return None
    
    logger.info(f"Configured repositories: {len(repos)}")
    for repo in repos:
        logger.info(f"  - {repo.name or repo.url}")
    
    async with aiohttp.ClientSession() as session:
        try:
            payload = {
                "skip_embeddings": False  # We want embeddings!
            }
            
            logger.info(f"POST {API_URL}/ingestion/trigger")
            logger.info(f"Payload: {json.dumps(payload, indent=2)}")
            
            async with session.post(
                f"{API_URL}/ingestion/trigger",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                result = await resp.json()
                
                if resp.status == 200:
                    job_id = result.get("job_id")
                    logger.info(f"✓ Ingestion triggered successfully")
                    logger.info(f"  Job ID: {job_id}")
                    logger.info(f"  Status URL: {result.get('status_url')}")
                    return job_id
                else:
                    logger.error(f"✗ Failed to trigger ingestion (HTTP {resp.status})")
                    logger.error(f"  Response: {result}")
                    return None
                    
        except aiohttp.ClientConnectorError:
            logger.error(f"✗ Could not connect to API at {API_URL}")
            logger.error("  Make sure the backend is running: python -m uvicorn src.api.main:app --reload")
            return None
        except Exception as e:
            logger.error(f"✗ Failed to trigger ingestion: {str(e)}", exc_info=True)
            return None


async def check_status(job_id: str, max_wait: int = 300):
    """Poll job status until completion"""
    logger.info("=" * 60)
    logger.info("MONITORING INGESTION PROGRESS")
    logger.info("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                logger.warning(f"⏱ Timeout after {max_wait}s - ingestion may still be running")
                return
            
            try:
                async with session.get(
                    f"{API_URL}/ingestion/status/{job_id}",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        status = await resp.json()
                        
                        current = status.get("current_step", "unknown")
                        progress = status.get("progress", 0)
                        job_status = status.get("status", "unknown")
                        stats = status.get("stats", {})
                        
                        logger.info(
                            f"[{int(elapsed)}s] Status: {job_status.upper()} | "
                            f"Progress: {progress:.0%} | Step: {current}"
                        )
                        logger.debug(f"  Stats: {stats}")
                        
                        if job_status == "completed":
                            logger.info("✓ Ingestion COMPLETED")
                            logger.info(f"  Final Stats:")
                            for key, value in stats.items():
                                logger.info(f"    - {key}: {value}")
                            return stats
                        elif job_status == "failed":
                            logger.error("✗ Ingestion FAILED")
                            errors = status.get("errors", [])
                            if errors:
                                logger.error("  Errors:")
                                for error in errors:
                                    logger.error(f"    - {error}")
                            return None
                        
                        # Wait before polling again
                        await asyncio.sleep(2)
                    else:
                        logger.warning(f"Status check returned HTTP {resp.status}")
                        await asyncio.sleep(2)
                        
            except aiohttp.ClientConnectorError:
                logger.warning(f"API temporarily unreachable, retrying...")
                await asyncio.sleep(2)
            except asyncio.TimeoutError:
                logger.warning(f"Status check timeout, retrying...")
                await asyncio.sleep(2)
            except Exception as e:
                logger.warning(f"Error checking status: {str(e)}, retrying...")
                await asyncio.sleep(2)


async def main():
    """Main test routine"""
    logger.info("Starting ingestion test...")
    logger.info("")
    
    # Trigger ingestion
    job_id = await trigger_ingestion()
    
    if not job_id:
        logger.error("✗ Failed to trigger ingestion")
        return 1
    
    logger.info("")
    
    # Monitor progress
    stats = await check_status(job_id)
    
    if stats:
        logger.info("")
        logger.info("✓ Ingestion test completed successfully!")
        return 0
    else:
        logger.error("")
        logger.error("✗ Ingestion test failed or timed out")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Ingestion test interrupted by user")
        sys.exit(1)
