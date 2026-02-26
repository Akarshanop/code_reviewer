"""
Configuration & Secret Scanner — Feature 4
Parses .env, application.yml, docker-compose.yml, package.json, pom.xml
and creates ConfigEntry nodes in the graph, linked to the services/files
that consume them.

Also flags potential secrets (API keys, tokens, passwords) so the QA agent
can verify they're not hardcoded in code.
"""
import re
import json
from pathlib import Path, PurePosixPath
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

from src.core.logger import get_logger
from src.intelligence.utils import generate_node_id

logger = get_logger(__name__)

# Patterns that indicate a value is likely a secret
SECRET_PATTERNS = [
    re.compile(r'(?i)(password|passwd|pwd)\s*[=:]\s*\S+'),
    re.compile(r'(?i)(secret|token|api[_-]?key|auth[_-]?key)\s*[=:]\s*\S+'),
    re.compile(r'(?i)(access[_-]?key|private[_-]?key)\s*[=:]\s*\S+'),
    re.compile(r'(?i)(credentials|connection[_-]?string)\s*[=:]\s*\S+'),
    # Looks like a JWT or long hex/base64 token
    re.compile(r'eyJ[A-Za-z0-9_-]{20,}'),
    re.compile(r'[A-Fa-f0-9]{32,}'),
]

# Config files we know how to parse
CONFIG_PATTERNS = {
    '.env': 'dotenv',
    '.env.local': 'dotenv',
    '.env.production': 'dotenv',
    '.env.development': 'dotenv',
    'docker-compose.yml': 'docker_compose',
    'docker-compose.yaml': 'docker_compose',
    'application.yml': 'spring_config',
    'application.yaml': 'spring_config',
    'application.properties': 'properties',
    'package.json': 'package_json',
    'requirements.txt': 'requirements',
    'pom.xml': 'pom',
    'Dockerfile': 'dockerfile',
}


class ConfigScanner:
    """
    Post-ingestion pass that:
    1. Finds config files in the repo file list
    2. Creates ConfigEntry nodes for each key-value pair
    3. Flags potential secrets
    4. Links config entries to code files that reference them
    """

    async def scan_configs(self, session, repo_hash: str, repo_path: str = "") -> Dict[str, Any]:
        stats = {
            "config_files_found": 0,
            "config_entries_created": 0,
            "secrets_flagged": 0,
            "config_usage_links": 0,
            "errors": [],
        }

        try:
            # ── 1. Find config files already in the graph ──
            file_result = await session.run("""
                MATCH (f:File)
                WHERE f.repository_hash = $repo_hash
                RETURN f.id AS id, f.path AS path
            """, {"repo_hash": repo_hash})

            records = [r async for r in file_result]
            config_files: List[Dict[str, Any]] = []
            code_files: List[Dict[str, Any]] = []

            for rec in records:
                path = rec["path"] or ""
                filename = PurePosixPath(path).name.lower()
                is_config = False
                for pattern_name, config_type in CONFIG_PATTERNS.items():
                    if filename == pattern_name.lower() or path.endswith(pattern_name):
                        config_files.append({
                            "id": rec["id"], "path": path,
                            "type": config_type, "filename": filename,
                        })
                        is_config = True
                        break
                if not is_config:
                    code_files.append(rec)

            stats["config_files_found"] = len(config_files)

            if not config_files:
                # Try to read .env and config files directly from repo_path if available
                if repo_path:
                    config_files = self._find_config_files_on_disk(repo_path, repo_hash)
                    stats["config_files_found"] = len(config_files)

            if not config_files:
                logger.info("No config files found")
                return stats

            # ── 2. Label config files ──
            config_file_ids = [cf["id"] for cf in config_files if cf.get("id")]
            if config_file_ids:
                await session.run("""
                    UNWIND $ids AS fid
                    MATCH (f:File {id: fid})
                    SET f:ConfigFile, f.is_config = true
                """, {"ids": config_file_ids})

            # ── 3. Parse each config file and extract entries ──
            all_entries: List[Dict[str, Any]] = []
            for cf in config_files:
                try:
                    entries = await self._parse_config_file(cf, repo_path)
                    all_entries.extend(entries)
                except Exception as e:
                    stats["errors"].append(f"Failed to parse {cf['path']}: {e}")

            # ── 4. Create ConfigEntry nodes ──
            if all_entries:
                for chunk_start in range(0, len(all_entries), 200):
                    chunk = all_entries[chunk_start:chunk_start + 200]
                    await session.run("""
                        UNWIND $entries AS e
                        MERGE (c:ConfigEntry {id: e.id})
                        SET c.key = e.key,
                            c.value = e.value,
                            c.file = e.file,
                            c.config_type = e.config_type,
                            c.is_secret = e.is_secret,
                            c.repository_hash = $repo_hash,
                            c.last_updated = datetime()
                        WITH c, e
                        MATCH (f:File {id: e.file_id})
                        MERGE (f)-[:DEFINES_CONFIG]->(c)
                    """, {"entries": chunk, "repo_hash": repo_hash})
                stats["config_entries_created"] = len(all_entries)
                stats["secrets_flagged"] = sum(1 for e in all_entries if e.get("is_secret"))

            # ── 5. Link config entries to code files that reference them ──
            # Find code files that use env var names (e.g., process.env.X, os.environ['X'])
            config_keys = list(set(e["key"] for e in all_entries if e.get("key")))
            if config_keys:
                usage_count = await self._link_config_usage(session, repo_hash, config_keys)
                stats["config_usage_links"] = usage_count

            logger.info(
                f"Config scan: {stats['config_files_found']} files, "
                f"{stats['config_entries_created']} entries, "
                f"{stats['secrets_flagged']} secrets flagged, "
                f"{stats['config_usage_links']} usage links"
            )

        except Exception as e:
            logger.error(f"Config scanning failed: {e}")
            stats["errors"].append(str(e))

        return stats

    def _find_config_files_on_disk(self, repo_path: str, repo_hash: str) -> List[Dict]:
        """Scan disk for config files if not in graph."""
        found = []
        rp = Path(repo_path)
        if not rp.exists():
            return found
        for pattern_name, config_type in CONFIG_PATTERNS.items():
            for match in rp.rglob(pattern_name):
                rel = str(match.relative_to(rp))
                found.append({
                    "id": None,
                    "path": rel,
                    "type": config_type,
                    "filename": match.name,
                    "disk_path": str(match),
                })
        return found

    async def _parse_config_file(self, cf: Dict, repo_path: str) -> List[Dict]:
        """Parse a config file and return a list of entry dicts."""
        entries = []
        path = cf["path"]
        config_type = cf["type"]
        file_id = cf.get("id", "")

        # Read content
        content = ""
        disk_path = cf.get("disk_path", "")
        if disk_path:
            try:
                with open(disk_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception:
                pass
        elif repo_path:
            try:
                full = Path(repo_path) / path
                with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception:
                pass

        if not content:
            return entries

        if config_type == 'dotenv':
            entries = self._parse_dotenv(content, path, file_id)
        elif config_type in ('docker_compose', 'spring_config'):
            entries = self._parse_yaml_config(content, path, file_id, config_type)
        elif config_type == 'properties':
            entries = self._parse_properties(content, path, file_id)
        elif config_type == 'package_json':
            entries = self._parse_package_json(content, path, file_id)

        return entries

    def _parse_dotenv(self, content: str, path: str, file_id: str) -> List[Dict]:
        entries = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                is_secret = self._is_secret(key, value)
                eid = generate_node_id("config", path, key)
                entries.append({
                    "id": eid, "key": key,
                    "value": "***REDACTED***" if is_secret else value[:200],
                    "file": path, "file_id": file_id,
                    "config_type": "dotenv", "is_secret": is_secret,
                })
        return entries

    def _parse_yaml_config(self, content: str, path: str, file_id: str, config_type: str) -> List[Dict]:
        entries = []
        try:
            import yaml
            data = yaml.safe_load(content)
            if isinstance(data, dict):
                self._flatten_dict(data, "", entries, path, file_id, config_type)
        except Exception:
            pass
        return entries

    def _flatten_dict(self, d: dict, prefix: str, entries: list, path: str, file_id: str, config_type: str):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                self._flatten_dict(v, full_key, entries, path, file_id, config_type)
            else:
                value_str = str(v) if v is not None else ""
                is_secret = self._is_secret(full_key, value_str)
                eid = generate_node_id("config", path, full_key)
                entries.append({
                    "id": eid, "key": full_key,
                    "value": "***REDACTED***" if is_secret else value_str[:200],
                    "file": path, "file_id": file_id,
                    "config_type": config_type, "is_secret": is_secret,
                })

    def _parse_properties(self, content: str, path: str, file_id: str) -> List[Dict]:
        entries = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()
                is_secret = self._is_secret(key, value)
                eid = generate_node_id("config", path, key)
                entries.append({
                    "id": eid, "key": key,
                    "value": "***REDACTED***" if is_secret else value[:200],
                    "file": path, "file_id": file_id,
                    "config_type": "properties", "is_secret": is_secret,
                })
        return entries

    def _parse_package_json(self, content: str, path: str, file_id: str) -> List[Dict]:
        entries = []
        try:
            data = json.loads(content)
            # Dependencies are config entries
            for dep_type in ("dependencies", "devDependencies", "peerDependencies"):
                deps = data.get(dep_type, {})
                if isinstance(deps, dict):
                    for pkg, ver in deps.items():
                        eid = generate_node_id("config", path, f"{dep_type}.{pkg}")
                        entries.append({
                            "id": eid, "key": f"{dep_type}.{pkg}",
                            "value": str(ver), "file": path, "file_id": file_id,
                            "config_type": "package_json", "is_secret": False,
                        })
            # Scripts
            for script_name, cmd in data.get("scripts", {}).items():
                eid = generate_node_id("config", path, f"scripts.{script_name}")
                entries.append({
                    "id": eid, "key": f"scripts.{script_name}",
                    "value": str(cmd)[:200], "file": path, "file_id": file_id,
                    "config_type": "package_json", "is_secret": False,
                })
        except Exception:
            pass
        return entries

    def _is_secret(self, key: str, value: str) -> bool:
        """Heuristic: does key or value look like a secret?"""
        key_lower = key.lower()
        secret_keywords = ('password', 'secret', 'token', 'api_key', 'apikey',
                          'auth_key', 'access_key', 'private_key', 'credentials',
                          'connection_string', 'jwt', 'bearer')
        if any(kw in key_lower for kw in secret_keywords):
            return True
        # Check value patterns
        for pattern in SECRET_PATTERNS:
            if pattern.search(value):
                return True
        return False

    async def _link_config_usage(self, session, repo_hash: str, config_keys: List[str]) -> int:
        """
        Link ConfigEntry nodes to code files that reference their keys.
        Uses a single UNWIND batch instead of one query per key.
        """
        keys = config_keys[:200]
        if not keys:
            return 0

        result = await session.run("""
            UNWIND $keys AS key
            MATCH (c:ConfigEntry {key: key, repository_hash: $repo_hash})
            MATCH (fn:Function)
            WHERE fn.code_body CONTAINS key
              AND fn.repository_hash = $repo_hash
            MERGE (fn)-[r:USES_CONFIG]->(c)
            ON CREATE SET r.created_at = datetime()
            SET r.last_updated = datetime()
            RETURN count(r) AS cnt
        """, {"keys": keys, "repo_hash": repo_hash})

        count = 0
        async for rec in result:
            count += rec.get("cnt", 0)
        return count
