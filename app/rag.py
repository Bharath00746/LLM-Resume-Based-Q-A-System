from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Generator, AsyncGenerator, Optional
import logging
import json

import google.generativeai as genai

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings as LISettings,
    Document,
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import LLM, LLMMetadata

from app.settings import settings
from app.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# ---------------- logging setup ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Gemini LLM wrapper ---------------- #
class GeminiLLM(LLM):
    model_config = {"protected_namespaces": ()}

    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None, **kwargs):
        super().__init__(**(kwargs or {}))
        if not api_key:
            raise RuntimeError("❌ GEMINI_API_KEY missing in .env")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self._model_name = model

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,
            num_output=1024,
            model_name=self._model_name
        )

    # ---------------- sync completions ----------------
    def complete(self, prompt: str, **kwargs) -> str:
        resp = self._model.generate_content(prompt)
        return getattr(resp, "text", str(resp)) or str(resp)

    async def acomplete(self, prompt: str, **kwargs) -> str:
        return self.complete(prompt, **kwargs)

    # ---------------- chat API ----------------
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        prompt = "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in messages])
        return self.complete(prompt, **kwargs)

    async def achat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        return self.chat(messages, **kwargs)

    # ---------------- streaming fallback ----------------
    def stream_complete(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        yield self.complete(prompt, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield self.complete(prompt, **kwargs)

    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        yield self.chat(messages, **kwargs)

    async def astream_chat(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        yield self.chat(messages, **kwargs)


# ---------------- Setup helpers ---------------- #
def _init_llm_and_embeddings():
    llm = GeminiLLM(model="gemini-1.5-flash", api_key=settings.GEMINI_API_KEY)
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
    LISettings.llm = llm
    LISettings.embed_model = embed_model
    LISettings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=120)
    return embed_model.model_name  # Return model name for consistency check


def _resume_path() -> Path:
    return Path(settings.RESUME_DIR) / settings.RESUME_FILENAME


def _storage_path() -> Path:
    return Path(settings.INDEX_DIR)


def build_or_load_index(force_rebuild: bool = False) -> VectorStoreIndex:
    embed_model_name = _init_llm_and_embeddings()
    storage_dir = _storage_path()

    # Check for existing index and embedding model consistency
    if storage_dir.exists() and not force_rebuild:
        try:
            storage_ctx = StorageContext.from_defaults(persist_dir=str(storage_dir))
            index = load_index_from_storage(storage_ctx)
            # Check if stored embedding model matches current model
            stored_metadata = storage_ctx.to_dict().get("metadata", {})
            stored_embed_model = stored_metadata.get("embed_model_name", "unknown")
            if stored_embed_model != embed_model_name:
                logger.warning(f"Embedding model mismatch: stored={stored_embed_model}, current={embed_model_name}. Rebuilding index.")
                force_rebuild = True
            else:
                logger.info(f"Loaded index with embedding model: {stored_embed_model}")
                return index
        except Exception as e:
            logger.warning(f"Failed to load index, rebuilding: {e}")
            force_rebuild = True

    if force_rebuild and storage_dir.exists():
        # Delete existing index to avoid dimension conflicts
        import shutil
        shutil.rmtree(storage_dir)
        logger.info(f"Deleted existing index at {storage_dir}")

    resume_file = _resume_path()
    if not resume_file.exists():
        raise FileNotFoundError(f"❌ Resume not found at {resume_file}")

    docs: List[Document] = SimpleDirectoryReader(input_files=[str(resume_file)]).load_data()
    logger.info(f"Loaded documents: {[doc.text[:100] for doc in docs]}")
    index = VectorStoreIndex.from_documents(docs)
    # Store embedding model name in index metadata
    index.storage_context.metadata = {"embed_model_name": embed_model_name}
    index.storage_context.persist(persist_dir=str(storage_dir))
    logger.info(f"✅ Index built and persisted at {storage_dir} with embedding model: {embed_model_name}")
    return index


def get_query_engine(index: VectorStoreIndex) -> RetrieverQueryEngine:
    retriever = VectorIndexRetriever(index=index, similarity_top_k=settings.TOP_K)
    synthesizer = get_response_synthesizer()
    return RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesizer)


def grounded_answer(
    index: VectorStoreIndex,
    question: str,
    fallback: str = None
) -> Dict[str, Any]:
    if fallback is None:
        fallback = """
        Bharath G is a Software Developer with 1 year of experience in Python and Django, based in Chennai, India. 
        He worked at Besant Technologies (Dec 2023 – Dec 2024), building web applications and projects like a 
        Library Management System and a Real-Time Chat Application. For more details, ask about his skills, 
        projects, or experience.
        """

    qe = get_query_engine(index)
    nodes = qe.retriever.retrieve(question)
    logger.info(f"Retrieved nodes for '{question}': {[n.node.get_content()[:100] for n in nodes]}")
    logger.info(f"Node scores: {[getattr(n, 'score', 0) for n in nodes]}")

    filtered_nodes = [n for n in nodes if getattr(n, "score", 1) >= 0.5]  # Lowered cutoff for better retrieval

    if not filtered_nodes:
        logger.info(f"No relevant chunks found for question: {question}")
        return {"answer": fallback, "from_resume": False, "sources": []}

    ctx_lines, sources = [], []
    for i, n in enumerate(filtered_nodes):
        meta = n.node.metadata or {}
        chunk_id = n.node.id_
        page = meta.get("page_label") or meta.get("page")
        doc = meta.get("file_name") or settings.RESUME_FILENAME
        ctx_lines.append(f"[Chunk {i+1} | {doc} | page={page}] {n.node.get_content()}")
        sources.append({"doc": doc, "page": page if page else None, "chunk_id": chunk_id})

    context = "\n\n".join(ctx_lines)
    user_prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    llm: GeminiLLM = LISettings.llm
    text = llm.complete(full_prompt).strip() or fallback

    logger.info(f"Question: {question}\nAnswer: {text}")
    return {"answer": text, "from_resume": True, "sources": sources}