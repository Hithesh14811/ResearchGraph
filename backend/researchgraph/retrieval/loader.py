"""LangChain document loading and chunking for retrieved sources.

``SourceDocumentLoader`` implements LangChain's ``BaseLoader`` interface over bytes we have
already fetched safely, so parsing/cleaning composes with any LangChain splitter or vector
store. Every chunk carries full provenance metadata.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from researchgraph.retrieval.cleaning import clean_text, compute_signals, split_references
from researchgraph.schemas.common import utc_now
from researchgraph.schemas.sources import ContentSignals, Source
from researchgraph.tools.documents import ExtractedDocument


def chunk_id(source_id: str, index: int) -> str:
    return f"{source_id}:{index}"


class SourceDocumentLoader(BaseLoader):
    """Turns one extracted source into a single cleaned LangChain ``Document``."""

    def __init__(self, *, research_id: str, source: Source, extracted: ExtractedDocument) -> None:
        self.research_id = research_id
        self.source = source
        self.extracted = extracted
        self.signals: ContentSignals | None = None

    def lazy_load(self) -> Iterator[Document]:
        body, references = split_references(clean_text(self.extracted.text))
        self.signals = compute_signals(body, references)
        published = self.source.published_date or self.extracted.published_date
        yield Document(
            page_content=body,
            metadata={
                "research_id": self.research_id,
                "source_id": self.source.id,
                "title": self.source.title,
                "url": self.source.url,
                "author": "; ".join(self.source.authors or self.extracted.authors)[:500],
                "date": published.isoformat() if published else "",
                "document_type": self.extracted.format.value,
                "provider": self.source.provider,
                "retrieved_at": utc_now().isoformat(),
            },
        )

    async def alazy_load(self) -> AsyncIterator[Document]:
        for document in self.lazy_load():
            yield document


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(chunk_overlap, chunk_size // 3),
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator="end",
    )


def chunk_document(document: Document, splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    """Split and enrich each chunk with its index and a stable ID."""
    chunks = splitter.split_documents([document])
    source_id = document.metadata["source_id"]
    enriched = []
    for index, chunk in enumerate(chunks):
        metadata = {**chunk.metadata, "chunk_index": index, "chunk_id": chunk_id(source_id, index)}
        enriched.append(
            Document(
                page_content=chunk.page_content, metadata=metadata, id=chunk_id(source_id, index)
            )
        )
    return enriched
