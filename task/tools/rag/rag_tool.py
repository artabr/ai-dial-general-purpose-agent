import json
from typing import Any

import faiss
import numpy as np
from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.tools.rag.document_cache import DocumentCache
from task.utils.dial_file_conent_extractor import DialFileContentExtractor

_SYSTEM_PROMPT = """You are a document Q&A assistant. Answer the user's question based ONLY on the provided document context.
If the answer is not found in the context, say so clearly.
Be concise and accurate. Quote relevant parts when helpful."""


class RagTool(BaseTool):
    """
    Performs semantic search on documents to find and answer questions based on relevant content.
    Supports: PDF, TXT, CSV, HTML.
    """

    def __init__(
        self, endpoint: str, deployment_name: str, document_cache: DocumentCache
    ):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.document_cache = document_cache

        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return (
            "Performs semantic search on a document to find relevant content and answer questions. "
            "Use this tool when the user asks questions about the content of a specific document (PDF, TXT, CSV, HTML). "
            "This tool indexes the document, searches for relevant chunks using semantic similarity, "
            "and generates an answer based on retrieved content. "
            "Provide the file URL and the specific question or search query. "
            "For large documents, this is more efficient than extracting all content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query or question to search for in the document",
                },
                "file_url": {
                    "type": "string",
                    "description": "The URL of the document to search in.",
                },
            },
            "required": ["request", "file_url"],
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        request = arguments["request"]
        file_url = arguments["file_url"]
        stage = tool_call_params.stage

        stage.append_content("## Request arguments: \n")
        stage.append_content(f"**Request**: {request}\n\r")
        stage.append_content(f"**File URL**: {file_url}\n\r")

        cache_document_key = f"{tool_call_params.conversation_id}:{file_url}"
        cached_data = self.document_cache.get(cache_document_key)

        if cached_data is not None:
            index, chunks = cached_data
        else:
            extractor = DialFileContentExtractor(
                endpoint=self.endpoint,
                api_key=tool_call_params.api_key,
            )
            text_content = extractor.extract_text(file_url)

            if not text_content:
                stage.append_content("File content not found.\n\r")
                return "Error: File content not found."

            chunks = self.text_splitter.split_text(text_content)
            embeddings = self.model.encode(chunks)
            index = faiss.IndexFlatL2(384)
            index.add(np.array(embeddings).astype('float32'))
            self.document_cache.set(cache_document_key, index, chunks)

        query_embedding = self.model.encode([request]).astype('float32')
        k = min(3, len(chunks))
        distances, indices = index.search(query_embedding, k=k)

        retrieved_chunks = [chunks[idx] for idx in indices[0]]

        augmented_prompt = self.__augmentation(request, retrieved_chunks)

        stage.append_content("## RAG Request: \n")
        stage.append_content(f"```text\n\r{augmented_prompt}\n\r```\n\r")
        stage.append_content("## Response: \n")

        client = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
            api_version="2025-01-01-preview",
        )

        chunks_response = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": augmented_prompt},
            ],
            stream=True,
        )

        collected_content = ""
        async for chunk in chunks_response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    tool_call_params.stage.append_content(delta.content)
                    collected_content += delta.content

        return collected_content

    def __augmentation(self, request: str, chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(chunks)
        return f"Context from document:\n\n{context}\n\n---\n\nQuestion: {request}"
