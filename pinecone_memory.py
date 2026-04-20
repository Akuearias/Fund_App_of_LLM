"""Pinecone-backed example memory for LLM judge few-shot retrieval.

This module provides a small, self-contained layer for:
1) embedding a task/submission/feedback tuple,
2) storing it in Pinecone,
3) retrieving semantically similar past examples,
4) formatting retrieved examples into a few-shot block.

Assumptions:
- You already created a Pinecone index.
- The index dimension matches the embedding model you choose.
- You have set the environment variables:
    - PINECONE_API_KEY
    - OPENAI_API_KEY

Recommended index settings for text-embedding-3-small:
- dimension: 1536
- metric: cosine
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pinecone import Pinecone

from dotenv import load_dotenv

load_dotenv()


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "locklm")
DEFAULT_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")


@dataclass
class MemoryExample:
    """A stored example that can be reused as a few-shot reference."""

    example_id: str
    goal: str
    submission: str
    rubric: str
    score: int
    feedback: str
    accepted: bool = True
    task_type: str = "general"
    rubric_version: str = "v1"
    namespace: str = DEFAULT_NAMESPACE

    def to_text(self) -> str:
        """Text used for embeddings."""
        return (
            f"GOAL:\n{self.goal}\n\n"
            f"RUBRIC:\n{self.rubric}\n\n"
            f"SUBMISSION:\n{self.submission}\n\n"
            f"SCORE:\n{self.score}\n\n"
            f"FEEDBACK:\n{self.feedback}"
        )

    def to_metadata(self) -> Dict[str, Any]:
        """Flat metadata only (Pinecone metadata should stay simple)."""
        return {
            "goal": self.goal,
            "submission": self.submission,
            "rubric": self.rubric,
            "score": self.score,
            "feedback": self.feedback,
            "accepted": self.accepted,
            "task_type": self.task_type,
            "rubric_version": self.rubric_version,
        }


class PineconeMemoryStore:
    """Small Pinecone wrapper for storing and retrieving judge examples."""

    def __init__(
        self,
        index_name: str = DEFAULT_INDEX_NAME,
        namespace: str = DEFAULT_NAMESPACE,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.index_name = index_name
        self.namespace = namespace
        self.embedding_model = embedding_model

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not pinecone_api_key:
            raise ValueError("Missing environment variable: PINECONE_API_KEY")
        if not openai_api_key:
            raise ValueError("Missing environment variable: OPENAI_API_KEY")

        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        self.client = OpenAI(api_key=openai_api_key)

    def embed_text(self, text: str) -> List[float]:
        """Create an embedding for text using OpenAI."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def store_example(self, example: MemoryExample) -> None:
        """Upsert one example into Pinecone."""
        vector = self.embed_text(example.to_text())
        self.index.upsert(
            vectors=[
                {
                    "id": example.example_id,
                    "values": vector,
                    "metadata": example.to_metadata(),
                }
            ],
            namespace=example.namespace,
        )

    def store_examples(self, examples: List[MemoryExample]) -> None:
        """Upsert multiple examples into Pinecone."""
        if not examples:
            return

        vectors = []
        for example in examples:
            vectors.append(
                {
                    "id": example.example_id,
                    "values": self.embed_text(example.to_text()),
                    "metadata": example.to_metadata(),
                }
            )

        # If you want a different namespace per example, split by namespace.
        self.index.upsert(vectors=vectors, namespace=examples[0].namespace)

    def retrieve_similar(
        self,
        goal: str,
        submission: str,
        top_k: int = 3,
        task_type: Optional[str] = None,
        rubric_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve semantically similar examples for few-shot prompting.

        Filtering is optional and uses Pinecone metadata filters.
        """
        query_text = f"GOAL:\n{goal}\n\nSUBMISSION:\n{submission}"
        query_vector = self.embed_text(query_text)

        filter_dict: Dict[str, Any] = {}
        if task_type:
            filter_dict["task_type"] = {"$eq": task_type}
        if rubric_version:
            filter_dict["rubric_version"] = {"$eq": rubric_version}

        query_kwargs: Dict[str, Any] = {
            "vector": query_vector,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": self.namespace,
        }
        if filter_dict:
            query_kwargs["filter"] = filter_dict

        result = self.index.query(**query_kwargs)
        matches = getattr(result, "matches", None)
        if matches is None:
            # Some SDK versions return a dict-like object.
            matches = result.get("matches", [])

        normalized: List[Dict[str, Any]] = []
        for match in matches:
            normalized.append(
                {
                    "id": match.get("id") if isinstance(match, dict) else match.id,
                    "score": match.get("score") if isinstance(match, dict) else match.score,
                    "metadata": match.get("metadata") if isinstance(match, dict) else match.metadata,
                }
            )
        return normalized

    @staticmethod
    def build_few_shot_block(matches: List[Dict[str, Any]], max_examples: int = 3) -> str:
        """Format retrieved matches into a compact few-shot prompt block."""
        if not matches:
            return ""

        snippets: List[str] = []
        for match in matches[:max_examples]:
            metadata = match.get("metadata", {}) or {}
            goal = metadata.get("goal", "")
            submission = metadata.get("submission", "")
            rubric = metadata.get("rubric", "")
            score = metadata.get("score", "")
            feedback = metadata.get("feedback", "")

            snippets.append(
                "\n".join(
                    [
                        "--- EXAMPLE ---",
                        f"GOAL: {goal}",
                        f"RUBRIC: {rubric}",
                        f"SUBMISSION: {submission}",
                        f"SCORE: {score}",
                        f"FEEDBACK: {feedback}",
                    ]
                )
            )

        return "\n\n".join(snippets)


def build_memory_store() -> PineconeMemoryStore:
    """Factory function used by the rest of the app."""
    return PineconeMemoryStore()


if __name__ == "__main__":
    # Minimal smoke test.
    store = build_memory_store()

    example = MemoryExample(
        example_id="seed-1",
        goal="Write a 500 word essay on Large Language Models",
        submission="LLMs are machine learning systems that...",
        rubric="Relevance, completeness, depth, clarity. Score strictly.",
        score=4,
        feedback="Strong structure and relevant content, but could include more technical detail.",
        accepted=True,
        task_type="essay",
        rubric_version="v1",
    )

    store.store_example(example)
    matches = store.retrieve_similar(
        goal=example.goal,
        submission=example.submission,
        top_k=3,
        task_type="essay",
        rubric_version="v1",
    )
    print(store.build_few_shot_block(matches))
