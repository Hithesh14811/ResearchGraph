from researchgraph.core.text import (
    coverage,
    extract_quantities,
    keywords,
    quote_in_text,
    sanitize_user_text,
    split_sentences,
)


def test_split_sentences_keeps_headings_separate_and_protects_abbreviations() -> None:
    text = "Results\nThe model scored 71.4% on average, e.g. on legal QA. Smith et al. agree. Next point!"
    assert split_sentences(text) == [
        "Results",
        "The model scored 71.4% on average, e.g. on legal QA.",
        "Smith et al. agree.",
        "Next point!",
    ]


def test_extract_quantities_ignores_years_and_small_counts() -> None:
    found = extract_quantities(
        "In 2024, 3 teams saw accuracy rise from 62% to 71.40% with 1,200 questions and 4.5x speedup."
    )
    assert found == {"62", "71.4", "1200", "4.5"}


def test_quote_matching_is_robust_to_case_whitespace_and_quote_style() -> None:
    source = "Fine-tuning was effective at teaching output style and domain terminology, improving accuracy."
    assert quote_in_text("fine-tuning  was effective at teaching OUTPUT style", source)
    assert quote_in_text(
        "“Fine-tuning was effective” at teaching output style",
        source.replace("Fine", "“Fine").replace("effective", "effective”"),
    )
    assert not quote_in_text(
        "fine-tuning was ineffective at teaching output style and domain terminology", source
    )


def test_quote_matching_allows_elisions() -> None:
    source = "Retrieval reduced errors by 30% on legal questions. It also lowered cost by half across all deployments."
    assert quote_in_text(
        "Retrieval reduced errors by 30% ... lowered cost by half across all deployments", source
    )


def test_keywords_are_unstemmed_and_stopword_free() -> None:
    assert (
        keywords("What practical guidance do credible sources give?", 5)
        == "practical guidance credible sources give"
    )


def test_coverage_measures_query_terms_found() -> None:
    assert (
        coverage("retrieval accuracy benchmarks", "Benchmark accuracy for retrieval systems") == 1.0
    )
    assert coverage("retrieval accuracy", "cooking recipes") == 0.0


def test_sanitize_user_text_strips_control_characters() -> None:
    assert (
        sanitize_user_text("Compare\x00 RAG\tand\n\nfine-tuning\x07", max_chars=100)
        == "Compare RAG and fine-tuning"
    )
    assert len(sanitize_user_text("x" * 500, max_chars=50)) == 50


def test_source_citation_markers_are_stripped_from_claims() -> None:
    """Regression (found on real arXiv papers): a paper's own "[14]" must never look like ours."""
    from researchgraph.core.text import strip_citation_markers

    text = "RAG grounds models [ 14], yet bias persists [3, 5] and drifts [12–15]. ALCE [16] helps."
    assert (
        strip_citation_markers(text)
        == "RAG grounds models, yet bias persists and drifts. ALCE helps."
    )
    assert (
        strip_citation_markers("Accuracy rose to 71.4% (n = 3,000).")
        == "Accuracy rose to 71.4% (n = 3,000)."
    )
