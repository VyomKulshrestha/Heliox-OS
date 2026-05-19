"""Workspace Index â€” local semantic search (RAG) for project files.

Uses FAISS for vector storage and sentence-transformers for embeddings.
Runs fully offline with no external API calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from pilot.utils.logger import get_logger

logger = get_logger("pilot.memory.workspace_index")

SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
}

MAX_FILE_SIZE_BYTES = 500_000
CHUNK_SIZE = 40


class WorkspaceIndex:
    """Local RAG engine for semantic search over workspace files."""

    def __init__(self, index_dir: Path) -> None:
        self._index_dir = index_dir
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = index_dir / "file_hashes.json"
        self._chunks_file = index_dir / "chunks.json"
        self._faiss_file = str(index_dir / "workspace.faiss")
        self._model = None
        self._index = None
        self._chunks: list[dict[str, Any]] = []
        self._file_hashes: dict[str, str] = {}
        self._ready = False

    def _load_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded: all-MiniLM-L6-v2")
            return True
        except ImportError:
            logger.warning("sentence-transformers not installed â€” workspace search disabled")
            return False
        except Exception:
            logger.exception("Failed to load embedding model")
            return False

    def _load_cache(self) -> None:
        if self._cache_file.exists():
            self._file_hashes = json.loads(self._cache_file.read_text())
        if self._chunks_file.exists():
            self._chunks = json.loads(self._chunks_file.read_text())

    def _save_cache(self) -> None:
        self._cache_file.write_text(json.dumps(self._file_hashes, indent=2))
        self._chunks_file.write_text(json.dumps(self._chunks, indent=2))

    def _file_hash(self, path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()

    def _chunk_file(self, path: Path, content: str) -> list[dict[str, Any]]:
        lines = content.splitlines()
        chunks = []
        for i in range(0, len(lines), CHUNK_SIZE // 2):
            chunk_lines = lines[i : i + CHUNK_SIZE]
            if not any(l.strip() for l in chunk_lines):
                continue
            chunks.append(
                {
                    "file": str(path),
                    "start_line": i + 1,
                    "end_line": i + len(chunk_lines),
                    "text": "\n".join(chunk_lines),
                }
            )
        return chunks

    def index_workspace(self, folder_path: str) -> dict[str, Any]:
        try:
            import faiss
            import numpy as np
        except ImportError:
            return {"success": False, "error": "faiss-cpu not installed. Run: pip install faiss-cpu"}

        if not self._load_model():
            return {"success": False, "error": "Embedding model unavailable"}

        self._load_cache()
        folder = Path(folder_path)
        if not folder.exists():
            return {"success": False, "error": f"Folder not found: {folder_path}"}

        new_chunks = []
        unchanged_chunks = []
        files_indexed = 0
        files_skipped = 0

        for file_path in folder.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix not in SUPPORTED_EXTENSIONS:
                continue
            if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                files_skipped += 1
                continue
            if any(
                p.startswith(".") or p in ("node_modules", "__pycache__", ".git", "venv", ".venv")
                for p in file_path.parts
            ):
                continue

            try:
                current_hash = self._file_hash(file_path)
                str_path = str(file_path)

                if self._file_hashes.get(str_path) == current_hash:
                    unchanged_chunks.extend(c for c in self._chunks if c["file"] == str_path)
                    continue

                content = file_path.read_text(encoding="utf-8", errors="ignore")
                file_chunks = self._chunk_file(file_path, content)
                new_chunks.extend(file_chunks)
                self._file_hashes[str_path] = current_hash
                files_indexed += 1

            except Exception as e:
                logger.debug("Skipping %s: %s", file_path, e)
                files_skipped += 1

        all_chunks = unchanged_chunks + new_chunks
        if not all_chunks:
            return {"success": True, "message": "No files to index", "files": 0}

        texts = [c["text"] for c in all_chunks]
        embeddings = self._model.encode(texts, show_progress_bar=False, batch_size=32)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        faiss.write_index(self._index, self._faiss_file)

        self._chunks = all_chunks
        self._save_cache()
        self._ready = True

        return {
            "success": True,
            "files_indexed": files_indexed,
            "files_unchanged": len(unchanged_chunks),
            "files_skipped": files_skipped,
            "total_chunks": len(all_chunks),
        }

    def search(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        try:
            import faiss
            import numpy as np
        except ImportError:
            return []

        if not self._load_model():
            return []

        if self._index is None:
            if not Path(self._faiss_file).exists():
                return []
            self._load_cache()
            self._index = faiss.read_index(self._faiss_file)
            self._ready = True

        if not self._chunks:
            return []

        query_vec = self._model.encode([query])
        query_vec = np.array(query_vec).astype("float32")
        faiss.normalize_L2(query_vec)

        k = min(n_results, len(self._chunks))
        scores, indices = self._index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                {
                    "file": chunk["file"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "score": float(score),
                    "text": chunk["text"],
                }
            )
        return results

    def is_ready(self) -> bool:
        return self._ready or Path(self._faiss_file).exists()
