"""Regression tests for mem0ai/mem0#7631 — the BM25 lemma token predicate.

The predicate decides which spaCy lemmas become BM25 terms. It used to be
``lemma.isalnum()``, which drops every hyphenated identifier because the
hyphen makes the string non-alphanumeric — and spaCy keeps ``gpt-4`` whole.
"""

import pytest

from mem0.utils.lemmatization import lemmatize_for_bm25


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I use GPT-4 and COVID-19 data", "gpt-4"),
        ("we migrated to mixtral-8x7b and gpt-4o", "mixtral-8x7b"),
        ("we index covid-19 case counts", "covid-19"),
    ],
)
def test_hyphenated_identifiers_survive(text, expected):
    """The reported bug: hyphenated model names never reach the index."""
    assert expected in lemmatize_for_bm25(text).split()


def test_query_side_matches_index_side():
    """Index and query must agree, or a stored term is unfindable.

    ``lemmatize_for_bm25`` lowercases before tokenizing, so a model name
    indexed from prose has to match the same name typed in a query.
    """
    indexed = lemmatize_for_bm25("we run GPT-4 in production").split()
    queried = lemmatize_for_bm25("gpt-4").split()

    assert "gpt-4" in indexed
    assert "gpt-4" in queried
    # Every term a query produces has to be findable in the stored text;
    # a term present in only one side makes the memory unfindable.
    assert set(queried) <= set(indexed), "query produced terms the index never stored"
    assert all(term == term.lower() for term in queried), "query side must be lowercased"


def test_bare_symbols_still_dropped():
    """Widening the predicate must not start indexing bare punctuation."""
    tokens = lemmatize_for_bm25("he said $ 7 @ home").split()
    assert "$" not in tokens
    assert "@" not in tokens
    assert "7" in tokens
    assert "home" in tokens


def test_plain_words_unaffected():
    """The ordinary path must not change: this is a targeted fix."""
    tokens = lemmatize_for_bm25("The meetings are attending quickly").split()
    assert "meeting" in tokens
    assert "attend" in tokens


def test_non_hyphen_punctuation_still_dropped():
    """The widening is hyphen-specific on purpose.

    An earlier revision accepted any lemma containing one alphanumeric
    character, which also let ``3.5``, ``node.js``, ``c++`` and
    ``user@example.com`` into the index. None of those are the reported
    defect, and admitting them widens the change well past a targeted fix.
    """
    tokens = lemmatize_for_bm25("ratio 1/2 on node.js and c++ at 3.5").split()

    for unwanted in ("1/2", "node.js", "c++", "3.5"):
        assert unwanted not in tokens, f"{unwanted!r} should stay out of the index"
