from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from qdrant_client.models import PointStruct

from src.core.id_generator import generate_node_id
from src.core.logger import get_logger
from src.database.qdrant_client import qdrant_client
from src.services.embedding_service import embedding_service

logger = get_logger(__name__)


class VectorEmbeddingIngestionBuilder:
    """
    Owns vector embedding ingestion for parsed repository artifacts.

    This is intentionally separate from runtime query/search concerns.
    """

    def __init__(self) -> None:
        self.qdrant = qdrant_client
        self.embedding = embedding_service

    async def initialize(self) -> None:
        await self.qdrant.initialize()
        await self.embedding.initialize()

    async def close(self) -> None:
        await self.qdrant.close()
        await self.embedding.close()

    def build_artifacts(
        self,
        parsed_data: Dict[str, Any],
        repo_url: str,
        repo_hash: str,
        repo_type: Optional[str] = None,
        entry_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        artifacts: List[Dict[str, Any]] = []
        repo_name = repo_url.split("/")[-1].replace(".git", "")

        # Extract artifacts from Java files.
        for java_file in parsed_data.get("java", []):
            file_data = java_file.get("data", {})
            file_path = java_file.get("file", "")
            file_name = file_path.split("/")[-1] if file_path else ""

            for route in file_data.get("routes", []):
                route_body = route.get("body", "") or ""
                route_summary = (
                    f"Route: {route.get('path', '')} "
                    f"Method: {route.get('method', '')} "
                    f"Handler: {route.get('handler_method', route.get('name', ''))}"
                )
                artifacts.append(
                    {
                        "type": "route",
                        "name": route.get("handler_method", route.get("name", "")),
                        "language": "java",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": route_body[:3000] if route_body else route_summary,
                        "text": route_summary,
                        "metadata": {
                            "path": route.get("path"),
                            "method": route.get("method"),
                            "handler": route.get("handler_method", route.get("name", "")),
                            "error_handling": route.get("error_handling", []),
                        },
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    }
                )

            for eb in file_data.get("eventbus", []):
                artifacts.append(
                    {
                        "type": "eventbus",
                        "name": eb.get("address", ""),
                        "language": "java",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": (
                            f"EventBus address: {eb.get('address', '')} "
                            f"type: {eb.get('type', '')} "
                            f"handler: {eb.get('handler_method', '')}"
                        ),
                        "text": (
                            f"EventBus: {eb.get('address', '')} "
                            f"Type: {eb.get('type', '')} "
                            f"Handler: {eb.get('handler_method', '')}"
                        ),
                        "metadata": {
                            "address": eb.get("address"),
                            "type": eb.get("type"),
                            "handler": eb.get("handler_method"),
                        },
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    }
                )

            for dto in file_data.get("dtos", []):
                artifacts.append(
                    {
                        "type": "dto",
                        "name": dto.get("name", ""),
                        "language": "java",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": (
                            f"DTO class {dto.get('name', '')} with fields: "
                            f"{', '.join([f.get('name', '') for f in dto.get('fields', [])])}"
                        ),
                        "text": (
                            f"DTO: {dto.get('name', '')} "
                            f"Fields: {', '.join([f.get('name', '') for f in dto.get('fields', [])])}"
                        ),
                        "metadata": {
                            "name": dto.get("name"),
                            "fields": dto.get("fields", []),
                            "is_data_object": dto.get("is_data_object", False),
                        },
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    }
                )

            for svc in file_data.get("services", []):
                svc_name = svc.get("name", "") if isinstance(svc, dict) else str(svc)
                svc_body = svc.get("body", "") if isinstance(svc, dict) else ""
                artifacts.append(
                    {
                        "type": "service",
                        "name": svc_name,
                        "language": "java",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": svc_body[:3000] if svc_body else f"Service class {svc_name} in {file_name}",
                        "text": f"Service: {svc_name} File: {file_name}",
                        "metadata": svc if isinstance(svc, dict) else {"name": svc_name},
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    }
                )

            for cls in file_data.get("classes", []):
                cls_name = cls.get("name", "") if isinstance(cls, dict) else str(cls)
                cls_body = cls.get("body", "") if isinstance(cls, dict) else ""
                methods_list = cls.get("methods", []) if isinstance(cls, dict) else []
                method_names = ", ".join(
                    [m.get("name", "") if isinstance(m, dict) else str(m) for m in methods_list[:10]]
                )
                artifacts.append(
                    {
                        "type": "class",
                        "name": cls_name,
                        "language": "java",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": (
                            cls_body[:3000]
                            if cls_body
                            else (
                                f"Class {cls_name} with methods: {method_names}"
                                if method_names
                                else f"Class {cls_name}"
                            )
                        ),
                        "text": (
                            f"Class: {cls_name} Methods: {method_names}"
                            if method_names
                            else f"Class: {cls_name}"
                        ),
                        "metadata": cls if isinstance(cls, dict) else {"name": cls_name},
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    }
                )

            for func in file_data.get("functions", []):
                if not isinstance(func, dict):
                    continue
                func_name = func.get("name", "")
                func_body = func.get("body", "")
                if not func_name:
                    continue
                artifacts.append(
                    {
                        "type": "function",
                        "name": func_name,
                        "language": "java",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": func_body[:3000] if func_body else f"Java method {func_name}",
                        "text": f"Java Method: {func_name} Returns: {func.get('return_type', 'void')}",
                        "metadata": {
                            "return_type": func.get("return_type"),
                            "parameters": func.get("parameters", []),
                            "calls": func.get("calls", []),
                            "error_handling": func.get("error_handling", []),
                        },
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    }
                )

        # Extract artifacts from Python files.
        for py_file in parsed_data.get("python", []):
            file_data = py_file.get("data", {})
            file_path = py_file.get("file", "")
            file_name = file_path.split("/")[-1] if file_path else ""

            for cls in file_data.get("classes", []):
                cls_name = cls.get("name", "") if isinstance(cls, dict) else str(cls)
                cls_body = cls.get("body", "") if isinstance(cls, dict) else ""
                methods_list = cls.get("methods", []) if isinstance(cls, dict) else []
                method_names = ", ".join(
                    [m.get("name", "") if isinstance(m, dict) else str(m) for m in methods_list[:10]]
                )
                artifacts.append(
                    {
                        "type": "class",
                        "name": cls_name,
                        "language": "python",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": (
                            cls_body[:3000]
                            if cls_body
                            else (
                                f"Python class {cls_name} with methods: {method_names}"
                                if method_names
                                else f"Python class {cls_name}"
                            )
                        ),
                        "text": (
                            f"Python Class: {cls_name} Methods: {method_names}"
                            if method_names
                            else f"Python Class: {cls_name}"
                        ),
                        "metadata": cls if isinstance(cls, dict) else {"name": cls_name},
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    }
                )

            for func in file_data.get("functions", []):
                if not isinstance(func, dict):
                    continue
                func_name = func.get("name", "")
                func_body = func.get("body", "")
                func_desc = func.get("docstring", "") or func.get("description", "")
                params = func.get("parameters", func.get("params", []))
                if isinstance(params, list):
                    param_parts = []
                    for p in params:
                        if isinstance(p, dict):
                            pname = p.get("name", "")
                            ptype = p.get("type", "")
                            param_parts.append(f"{pname}: {ptype}" if ptype else pname)
                        else:
                            param_parts.append(str(p))
                    param_str = ", ".join(param_parts)
                else:
                    param_str = str(params)
                artifacts.append(
                    {
                        "type": "function",
                        "name": func_name,
                        "language": "python",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": func_body[:3000] if func_body else f"def {func_name}({param_str})",
                        "text": f"Python Function: {func_name}({param_str}) {func_desc[:200]}",
                        "metadata": {
                            "parameters": params,
                            "return_type": func.get("return_type"),
                            "calls": func.get("calls", []),
                            "error_handling": func.get("error_handling", []),
                            "is_async": func.get("is_async", False),
                        },
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    }
                )

            for route in file_data.get("routes", []):
                route_path = route.get("path", "") if isinstance(route, dict) else str(route)
                method = route.get("method", "GET") if isinstance(route, dict) else "GET"
                handler = route.get("handler", "") if isinstance(route, dict) else ""
                artifacts.append(
                    {
                        "type": "route",
                        "name": handler or route_path,
                        "language": "python",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": f"Python Route: {method} {route_path} Handler: {handler}",
                        "text": f"Python Route: {method} {route_path} Handler: {handler}",
                        "metadata": route if isinstance(route, dict) else {"path": route_path},
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    }
                )

        # Extract artifacts from TypeScript files.
        for ts_file in parsed_data.get("typescript", []):
            file_data = ts_file.get("data", {})
            file_metadata = ts_file.get("metadata", {})
            file_path = ts_file.get("file", "")
            file_name = file_path.split("/")[-1] if file_path else ""

            components = file_data.get("components", []) or file_metadata.get("components", [])
            for comp in components:
                comp_body = comp.get("body", "") or ""
                artifacts.append(
                    {
                        "type": "react_component",
                        "name": comp.get("name", ""),
                        "language": "typescript",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": comp_body[:3000] if comp_body else f"React component {comp.get('name', '')}",
                        "text": (
                            f"React Component: {comp.get('name', '')} "
                            f"Props: {', '.join([p.get('name', '') for p in comp.get('props', [])])} "
                            f"Hooks: {', '.join(comp.get('hooks', []))}"
                        ),
                        "metadata": {
                            "name": comp.get("name"),
                            "props": comp.get("props", []),
                            "hooks": comp.get("hooks", []),
                        },
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    }
                )

            for func in file_data.get("functions", []):
                func_body = func.get("body", "") or ""
                artifacts.append(
                    {
                        "type": "function",
                        "name": func.get("name", ""),
                        "language": "typescript",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": func_body[:3000] if func_body else f"Function {func.get('name', '')}",
                        "text": f"TS Function: {func.get('name', '')} calls: {', '.join(func.get('calls', [])[:5])}",
                        "metadata": {
                            "name": func.get("name"),
                            "calls": func.get("calls", []),
                            "is_async": func.get("is_async", False),
                        },
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    }
                )

            for api in file_data.get("api_calls", []):
                artifacts.append(
                    {
                        "type": "api_call",
                        "name": api.get("url", ""),
                        "language": "typescript",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": (
                            f"API call to {api.get('url', '')} using {api.get('method', '')} "
                            f"in hook {api.get('hook', '')}"
                        ),
                        "text": (
                            f"API Call: {api.get('url', '')} "
                            f"Method: {api.get('method', '')} "
                            f"Hook: {api.get('hook', '')}"
                        ),
                        "metadata": {
                            "url": api.get("url"),
                            "method": api.get("method"),
                            "hook": api.get("hook"),
                            "component": api.get("component"),
                        },
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    }
                )

        # Extract artifacts from JavaScript files.
        for js_file in parsed_data.get("javascript", []):
            file_data = js_file.get("data", {})
            file_metadata = js_file.get("metadata", {})
            file_path = js_file.get("file", "")
            file_name = file_path.split("/")[-1] if file_path else ""

            components = file_data.get("components", []) or file_metadata.get("components", [])
            for comp in components:
                comp_body = comp.get("body", "") or ""
                artifacts.append(
                    {
                        "type": "react_component",
                        "name": comp.get("name", ""),
                        "language": "javascript",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": comp_body[:3000] if comp_body else f"React component {comp.get('name', '')}",
                        "text": (
                            f"React Component: {comp.get('name', '')} "
                            f"Props: {', '.join([p.get('name', '') for p in comp.get('props', [])])}"
                        ),
                        "metadata": {
                            "name": comp.get("name"),
                            "props": comp.get("props", []),
                        },
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    }
                )

            for func in file_data.get("functions", []):
                func_body = func.get("body", "") or ""
                artifacts.append(
                    {
                        "type": "function",
                        "name": func.get("name", ""),
                        "language": "javascript",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": func_body[:3000] if func_body else f"Function {func.get('name', '')}",
                        "text": f"JS Function: {func.get('name', '')} calls: {', '.join(func.get('calls', [])[:5])}",
                        "metadata": {
                            "name": func.get("name"),
                            "calls": func.get("calls", []),
                            "is_async": func.get("is_async", False),
                        },
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    }
                )

            for api in file_data.get("api_calls", []):
                artifacts.append(
                    {
                        "type": "api_call",
                        "name": api.get("url", ""),
                        "language": "javascript",
                        "repository": repo_url,
                        "repository_hash": repo_hash,
                        "file": file_path,
                        "code": f"API call to {api.get('url', '')} using {api.get('method', '')}",
                        "text": f"API Call: {api.get('url', '')} Method: {api.get('method', '')}",
                        "metadata": {
                            "url": api.get("url"),
                            "method": api.get("method"),
                        },
                        "repository_name": repo_name,
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": entry_file or "",
                        "repo_type": repo_type or "",
                    }
                )

        for artifact in artifacts:
            artifact["graph_node_id"] = self._compute_graph_node_id(
                artifact=artifact,
                repo_hash=repo_hash,
            )

        return artifacts

    async def index_code_artifacts(
        self, artifacts: List[Dict[str, Any]], batch_size: int = 100
    ) -> Dict[str, Any]:
        if not artifacts:
            logger.warning("No artifacts to index")
            return {"indexed": 0, "errors": []}

        total_indexed = 0
        errors: List[str] = []

        logger.info(
            "Starting artifact indexing",
            total_artifacts=len(artifacts),
            batch_size=batch_size,
        )

        for i in range(0, len(artifacts), batch_size):
            batch = artifacts[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            try:
                texts: List[str] = []
                metadata_list: List[Dict[str, Any]] = []
                artifact_names: List[str] = []

                for artifact in batch:
                    text = self._create_index_text(artifact)
                    if not text or len(text.strip()) < 5:
                        logger.warning(
                            "Artifact has minimal content",
                            artifact_type=artifact.get("type"),
                            artifact_name=artifact.get("name"),
                            text_length=len(text) if text else 0,
                        )

                    texts.append(text)
                    artifact_names.append(artifact.get("name", "unknown"))

                    file_path = artifact.get("file", "")
                    file_name = file_path.split("/")[-1] if file_path else ""

                    metadata = {
                        "type": artifact.get("type", "unknown"),
                        "language": artifact.get("language", "unknown"),
                        "repository": artifact.get("repository", "unknown"),
                        "repository_hash": artifact.get("repository_hash", ""),
                        "file": file_path,
                        "line": artifact.get("line", 0),
                        "name": artifact.get("name", ""),
                        "signature": artifact.get("signature", ""),
                        "description": artifact.get("description", ""),
                        "metadata": artifact.get("metadata", {}),
                        "created_at": datetime.utcnow().isoformat(),
                        "repository_name": artifact.get("repository_name", ""),
                        "file_name": file_name,
                        "relative_path": file_path,
                        "entry_file": artifact.get("entry_file", ""),
                        "repo_type": artifact.get("repo_type", ""),
                        "graph_node_id": artifact.get("graph_node_id", ""),
                    }
                    metadata_list.append(metadata)

                points_data = await self.embedding.batch_generate_with_ids(texts, metadata_list)
                if not points_data:
                    error_msg = f"Batch {batch_num}: Failed to generate embeddings (empty result)"
                    logger.error(error_msg, batch_index=batch_num)
                    errors.append(error_msg)
                    continue

                points: List[PointStruct] = []
                for point_data in points_data:
                    points.append(
                        PointStruct(
                            id=point_data["id"],
                            vector=point_data["vector"],
                            payload=point_data["payload"],
                        )
                    )

                if not points:
                    error_msg = f"Batch {batch_num}: No valid points to upsert"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue

                await self.qdrant.upsert_vectors(points)
                total_indexed += len(batch)
                logger.info(
                    "Indexed batch successfully",
                    batch_num=batch_num,
                    batch_size=len(batch),
                    total_indexed=total_indexed,
                    batch_artifacts=artifact_names[:3],
                )

            except Exception as e:
                error_msg = f"Batch {batch_num} failed: {str(e)}"
                logger.error(
                    "Failed to index batch",
                    batch_num=batch_num,
                    error=str(e),
                    exc_info=True,
                )
                errors.append(error_msg)

        return {
            "indexed": total_indexed,
            "errors": errors,
            "success": len(errors) == 0 and total_indexed > 0,
        }

    def summarize_artifact_types(self, artifacts: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for artifact in artifacts:
            counts[artifact.get("type", "unknown")] += 1
        return dict(counts)

    def _create_index_text(self, artifact: Dict[str, Any]) -> str:
        parts = []
        artifact_type = artifact.get("type", "artifact")
        artifact_name = artifact.get("name", "")
        parts.append(f"{artifact_type}: {artifact_name}")

        if artifact.get("language"):
            parts.append(f"Language: {artifact['language']}")
        if artifact.get("file"):
            parts.append(f"File: {artifact['file']}")
        if artifact.get("signature"):
            parts.append(f"Signature: {artifact['signature']}")
        if artifact.get("description"):
            parts.append(f"Description: {artifact['description']}")

        text_field = artifact.get("text") or ""
        code_field = artifact.get("code") or ""
        if text_field and code_field and text_field != code_field:
            combined = f"{text_field}\n{code_field}"
        else:
            combined = text_field or code_field

        if combined:
            if len(combined) > 2500:
                combined = combined[:2500] + "..."
            parts.append(combined)

        if artifact.get("metadata") and isinstance(artifact["metadata"], dict):
            for key, value in artifact["metadata"].items():
                if isinstance(value, str) and len(value) < 100:
                    parts.append(f"{key}: {value}")

        return "\n".join(parts)

    @staticmethod
    def _generate_graph_node_id(node_type: str, *identifiers: Any) -> str:
        return generate_node_id(node_type, *identifiers)

    def _compute_graph_node_id(self, artifact: Dict[str, Any], repo_hash: str) -> str:
        artifact_type = (artifact.get("type") or "").lower()
        file_path = artifact.get("file", "")
        metadata = artifact.get("metadata", {}) if isinstance(artifact.get("metadata"), dict) else {}

        if artifact_type == "route":
            path = metadata.get("path") or artifact.get("name", "")
            method = metadata.get("method") or "GET"
            return self._generate_graph_node_id("Route", repo_hash, path, method, file_path)
        if artifact_type == "eventbus":
            address = metadata.get("address") or artifact.get("name", "")
            eb_type = metadata.get("type") or "unknown"
            return self._generate_graph_node_id("EventBus", repo_hash, address, eb_type, file_path)
        if artifact_type == "dto":
            return self._generate_graph_node_id("DTO", repo_hash, artifact.get("name", ""), file_path)
        if artifact_type == "service":
            return self._generate_graph_node_id("Service", repo_hash, artifact.get("name", ""), file_path)
        if artifact_type == "react_component":
            return self._generate_graph_node_id("Component", repo_hash, artifact.get("name", ""), file_path)
        if artifact_type == "api_call":
            url = metadata.get("url") or artifact.get("name", "")
            method = metadata.get("method") or "GET"
            return self._generate_graph_node_id("ApiCall", repo_hash, url, method, file_path)
        if artifact_type == "function":
            return self._generate_graph_node_id("Function", repo_hash, file_path, artifact.get("name", ""))
        if artifact_type == "class":
            return self._generate_graph_node_id("Class", repo_hash, artifact.get("name", ""), file_path)
        return self._generate_graph_node_id("File", repo_hash, file_path)
