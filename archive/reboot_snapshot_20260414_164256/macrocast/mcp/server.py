"""FastMCP server exposing the Coulombe knowledge base via stdio transport.

Tools
-----
search_papers   Search paper text and the methodology document.
search_blog     Search blog posts for accessible explanations.
get_section     Retrieve a specific paper section by key + section name.
cite            Find evidence in the corpus supporting a claim.
ask_coulombe    General methodology question combining all source types.

Run:
    uv run python macrocast/mcp/server.py
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from mcp.server.fastmcp import FastMCP

from macrocast.mcp.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    PAPER_METADATA,
    make_embedding_function,
)

logger = logging.getLogger(__name__)

mcp = FastMCP("coulombe-kb")

# ---------------------------------------------------------------------------
# Lazy-initialised collection (avoids loading the model at import time)
# ---------------------------------------------------------------------------

_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        if not CHROMA_DIR.exists():
            raise RuntimeError(
                f"ChromaDB not found at {CHROMA_DIR}. "
                "Run `uv run python macrocast/mcp/indexer.py` first."
            )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        ef = make_embedding_function()
        _collection = client.get_collection(
            name=CHROMA_COLLECTION, embedding_function=ef
        )
    return _collection


# ---------------------------------------------------------------------------
# Helper: format a ChromaDB result dict into readable text
# ---------------------------------------------------------------------------


def _format_results(results: dict[str, Any]) -> str:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        return "No results found."

    parts = []
    for doc, meta, dist in zip(docs, metas, distances):
        source_type = meta.get("source_type", "unknown")
        section = meta.get("section", "")
        key = meta.get("paper_key", "")
        similarity = 1.0 - float(dist)

        if source_type == "paper":
            paper_info = PAPER_METADATA.get(key, {})
            header = (
                f"[{key}] {paper_info.get('title', '')} "
                f"({paper_info.get('authors', '')}, {paper_info.get('year', '')}) "
                f"— §{section}  [sim={similarity:.3f}]"
            )
        elif source_type == "blog":
            blog_date = meta.get("blog_date", "")
            header = f"[Blog] {section} ({blog_date})  [sim={similarity:.3f}]"
        else:
            header = f"[Methodology] §{section}  [sim={similarity:.3f}]"

        parts.append(f"{header}\n{doc}\n")

    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def search_papers(query: str, max_results: int = 5) -> str:
    """Search Coulombe's papers and the methodology document for a topic.

    Parameters
    ----------
    query:
        Natural-language question or keyword string.
    max_results:
        Number of chunks to return (default 5).

    Returns
    -------
    Formatted text of the most relevant passages with citation headers.
    """
    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=max_results,
        where={"source_type": {"$in": ["paper", "methodology"]}},
    )
    return _format_results(results)


@mcp.tool()
async def search_blog(query: str, max_results: int = 3) -> str:
    """Search Coulombe's blog posts for accessible explanations.

    Parameters
    ----------
    query:
        Natural-language question or keyword string.
    max_results:
        Number of chunks to return (default 3).

    Returns
    -------
    Formatted text of the most relevant blog passages.
    """
    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=max_results,
        where={"source_type": {"$eq": "blog"}},
    )
    return _format_results(results)


@mcp.tool()
async def get_section(paper_key: str, section: str) -> str:
    """Retrieve a specific section from a paper.

    Parameters
    ----------
    paper_key:
        Short paper key, e.g. ``CLSS2022``, ``C2024tvp``.
    section:
        Section name or keyword, e.g. ``Introduction``, ``Model``, ``Results``.

    Returns
    -------
    All chunks from that paper whose section name contains `section` (case-insensitive).
    """
    collection = _get_collection()
    # Fetch all chunks for this paper, then filter by section substring
    results = collection.get(
        where={"paper_key": {"$eq": paper_key}},
        include=["documents", "metadatas"],
    )
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])

    matched = [
        (doc, meta)
        for doc, meta in zip(docs, metas)
        if section.lower() in meta.get("section", "").lower()
    ]

    if not matched:
        available = sorted({m.get("section", "") for m in metas})
        return (
            f"No section matching '{section}' found in {paper_key}.\n"
            f"Available sections: {available}"
        )

    parts = []
    for doc, meta in matched:
        sec_name = meta.get("section", "")
        parts.append(f"[{paper_key}] §{sec_name}\n{doc}")
    return "\n---\n".join(parts)


@mcp.tool()
async def cite(claim: str) -> str:
    """Find supporting evidence in the Coulombe corpus for a claim.

    Parameters
    ----------
    claim:
        A statement you want to support with primary source evidence.

    Returns
    -------
    Up to 5 relevant passages from papers, blog, and methodology docs,
    with citation headers suitable for inline academic use.
    """
    collection = _get_collection()
    results = collection.query(
        query_texts=[claim],
        n_results=5,
    )
    return _format_results(results)


@mcp.tool()
async def ask_coulombe(question: str) -> str:
    """Answer a methodology or design question by retrieving context from the full corpus.

    Combines paper text, blog posts, and the methodology document.
    Use for design decisions in the Pipeline layer, interpretation questions,
    or any question grounded in Coulombe's research framework.

    Parameters
    ----------
    question:
        A natural-language question about methodology, design, or interpretation.

    Returns
    -------
    Up to 7 relevant passages from the full corpus.
    """
    collection = _get_collection()
    results = collection.query(
        query_texts=[question],
        n_results=7,
    )
    return _format_results(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    mcp.run(transport="stdio")
