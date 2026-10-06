import pytest

from researchgraph.core.errors import DocumentExtractionError
from researchgraph.demo.pdf import build_pdf
from researchgraph.retrieval.cleaning import clean_text, compute_signals, split_references
from researchgraph.retrieval.loader import build_splitter, chunk_document
from researchgraph.schemas.sources import DocumentFormat
from researchgraph.tools.documents import detect_format, extract_document, parse_frontmatter
from researchgraph.tools.metadata import extract_doi, parse_date

HTML = b"""<!doctype html><html><head><title>Fallback title</title>
<meta name="citation_title" content="A Study of Retrieval">
<meta name="citation_author" content="Ada Lovelace"><meta name="citation_author" content="Alan Turing">
<meta name="citation_publication_date" content="2024/05/17"></head>
<body><nav>Home | Subscribe to our newsletter</nav><article><h1>A Study of Retrieval</h1>
<p>Retrieval improved accuracy by 9 points.</p><script>alert(1)</script></article>
<footer>Accept all cookies</footer></body></html>"""


def _extract(body: bytes, content_type: str, url: str = "https://example.org/doc") -> object:
    return extract_document(
        body, content_type=content_type, url=url, max_chars=10_000, max_pdf_pages=5
    )


def test_pdf_extraction_roundtrip() -> None:
    pdf = build_pdf(
        "Abstract\nRetrieval improved exact match to 71.4% on the benchmark.",
        title="Paper",
        author="A. Author",
    )
    doc = _extract(pdf, "application/pdf")
    assert doc.format is DocumentFormat.PDF  # type: ignore[attr-defined]
    assert "71.4%" in doc.text  # type: ignore[attr-defined]
    assert doc.title == "Paper"  # type: ignore[attr-defined]


def test_malformed_pdf_raises_typed_error() -> None:
    with pytest.raises(DocumentExtractionError):
        _extract(b"%PDF-1.4 this is not really a pdf", "application/pdf")


def test_html_extraction_uses_citation_metadata_and_strips_boilerplate() -> None:
    doc = _extract(HTML, "text/html; charset=utf-8")
    assert doc.title == "A Study of Retrieval"  # type: ignore[attr-defined]
    assert doc.authors == ["Ada Lovelace", "Alan Turing"]  # type: ignore[attr-defined]
    assert str(doc.published_date) == "2024-05-17"  # type: ignore[attr-defined]
    assert "Retrieval improved accuracy by 9 points." in doc.text  # type: ignore[attr-defined]
    assert "alert" not in doc.text and "Subscribe" not in doc.text  # type: ignore[attr-defined]


def test_markdown_frontmatter_and_text() -> None:
    body = b"---\ntitle: My Doc\nauthors: A; B\ndate: 2025-01-02\n---\n# Heading\nSome **bold** [link](http://x.y) text."
    doc = _extract(body, "text/markdown")
    assert doc.title == "My Doc" and doc.authors == ["A", "B"]  # type: ignore[attr-defined]
    assert "Some bold link text." in doc.text  # type: ignore[attr-defined]
    meta, rest = parse_frontmatter("no front matter")
    assert meta == {} and rest == "no front matter"


def test_unsupported_and_empty_documents() -> None:
    with pytest.raises(DocumentExtractionError):
        _extract(b"binary", "application/octet-stream", url="https://example.org/file.bin")
    with pytest.raises(DocumentExtractionError):
        _extract(b"   ", "text/plain")
    assert detect_format(None, "https://x.org/paper.PDF?download=1") is DocumentFormat.PDF


def test_cleaning_reflows_pdf_lines_and_splits_references() -> None:
    body = (
        "Results\nThe model achieved high accu-\nracy on the test\nset.\n\n"
        + "We evaluate on three benchmarks with strong baselines.\n" * 8
    )
    raw = body + "\nReferences\n[1] A. Author et al. 2020. Title.\n[2] B. Author. 2021."
    cleaned = clean_text(raw)
    assert "high accuracy on the test set." in cleaned
    body, refs = split_references(cleaned)
    assert "References" not in body and "[1]" in refs
    signals = compute_signals(body, refs)
    assert signals.reference_count >= 2 and signals.word_count > 5


def test_chunking_assigns_stable_ids_and_metadata() -> None:
    from langchain_core.documents import Document

    doc = Document(
        page_content="Sentence about retrieval. " * 200,
        metadata={"source_id": "S-abc", "research_id": "r1"},
    )
    chunks = chunk_document(doc, build_splitter(400, 50))
    assert len(chunks) > 3
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert chunks[2].id == "S-abc:2" and chunks[2].metadata["research_id"] == "r1"


def test_metadata_helpers() -> None:
    assert str(parse_date("D:20230115120000Z")) == "2023-01-15"
    assert str(parse_date("March 3, 2021")) == "2021-03-03"
    assert parse_date("not a date") is None
    assert extract_doi("see https://doi.org/10.1234/abc.567 for details") == "10.1234/abc.567"


def test_titles_and_section_headings_never_fuse_with_sentences() -> None:
    """Regression: PDF extraction drops blank lines, which fused 'Title Abstract We compare…'."""
    from researchgraph.core.text import split_sentences

    pdf = build_pdf(
        "Retrieval, Fine-Tuning or Long Context? A Controlled Comparison on Domain-Specific QA\n\n"
        "Abstract\n\nWe compare retrieval and fine-tuning on three benchmarks.\n\nResults\n\nRetrieval won.",
        title="t",
    )
    text = _extract(pdf, "application/pdf").text  # type: ignore[attr-defined]
    sentences = split_sentences(clean_text(text))
    assert "We compare retrieval and fine-tuning on three benchmarks." in sentences
    assert "Abstract" in sentences and "Results" in sentences
    assert not any("QA Abstract" in s for s in sentences)


def test_dehyphenation_keeps_real_compounds_and_joins_syllable_breaks() -> None:
    from researchgraph.retrieval.cleaning import dehyphenate

    text = "Long-context prompting is costly. Long-\ncontext windows help. High accu-\nracy."
    assert (
        dehyphenate(text)
        == "Long-context prompting is costly. Long-context windows help. High accuracy."
    )


def test_html_headings_never_fuse_with_the_following_paragraph() -> None:
    """Regression: a 9-word <h1> was reflowed into the first sentence of the next <p>."""
    from researchgraph.core.text import split_sentences

    html = (
        b"<html><body><h1>Cascading Small and Large Models: A Production Case Study</h1>"
        b"<p>Routing each request to a small model first handled 78% of requests.</p>"
        b"<ul><li>First point</li><li>Second point</li></ul></body></html>"
    )
    text = _extract(html, "text/html").text  # type: ignore[attr-defined]
    sentences = split_sentences(clean_text(text))
    assert "Routing each request to a small model first handled 78% of requests." in sentences
    assert not any("Case Study Routing" in s for s in sentences)
    assert not any("First point Second point" in s for s in sentences)
