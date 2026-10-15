import pytest


@pytest.fixture(autouse=True)
def _ensure_spacy():
    """Skip tests if spaCy model is not available."""
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except Exception:
        pytest.skip("spaCy en_core_web_sm model not available")


class TestLemmatizeForBm25:
    def test_basic_lemmatization(self):
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("The cats are running quickly")
        assert "cat" in result
        assert "run" in result or "running" in result
        # Stop words and punctuation should be removed
        assert "the" not in result.split()

    def test_verb_forms_normalized(self):
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("she attended multiple meetings yesterday")
        assert "attend" in result or "attended" in result
        assert "meeting" in result  # -ing form preserved alongside lemma
        # "multiple" is kept (not a spaCy stop word)

    def test_ing_preservation(self):
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("attending the morning meeting")
        tokens = result.split()
        # Should have both the lemma and the -ing form
        assert "attending" in tokens or "attend" in tokens

    def test_empty_string(self):
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("")
        assert result == ""

    def test_punctuation_removed(self):
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_lowercased(self):
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("PYTHON Programming LANGUAGE")
        for token in result.split():
            assert token == token.lower()

    def test_stop_words_removed(self):
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("this is a very simple test of the system")
        tokens = result.split()
        for stop in ["this", "is", "a", "very", "of", "the"]:
            assert stop not in tokens

    def test_model_names_and_version_numbers_kept(self):
        """Model names carry a hyphen, so their lemma is not ``str.isalnum``.

        Dropping them removes the highest-frequency search terms from a memory
        store about model usage. See mem0ai/mem0#7631.
        """
        from mem0.utils.lemmatization import lemmatize_for_bm25

        tokens = lemmatize_for_bm25("I use GPT-4 and COVID-19 data").split()

        assert "gpt-4" in tokens
        assert "covid-19" in tokens
        assert "datum" in tokens

    def test_hyphenated_identifiers_kept(self):
        from mem0.utils.lemmatization import lemmatize_for_bm25

        tokens = lemmatize_for_bm25("we migrated to mixtral-8x7b and gpt-4o").split()

        assert "mixtral-8x7b" in tokens
        assert "gpt-4o" in tokens

    def test_symbol_only_tokens_still_removed(self):
        """Keeping hyphenated terms must not become keeping bare symbols.

        ``str.isalnum()`` was the only thing rejecting a lemma such as ``$``,
        which spaCy reports as a non-punctuation token, so the fix keeps a term
        only when it carries at least one alphanumeric character.
        """
        from mem0.utils.lemmatization import lemmatize_for_bm25

        tokens = lemmatize_for_bm25("he said $ 50 # 7 @ home").split()

        assert "$" not in tokens
        assert "50" in tokens
        assert "7" in tokens
        assert "home" in tokens
