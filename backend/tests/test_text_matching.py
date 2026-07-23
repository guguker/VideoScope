from videoscope.search.text_matching import SearchLexicon, lexical_match


def test_russian_inflection_matches_a_surname() -> None:
    match = lexical_match("Мозгов", "Сегодня Мозгова заменили в третьей четверти")

    assert match.score >= 0.9
    assert match.matched_terms == ("мозгов",)


def test_latin_transliteration_matches_cyrillic_name() -> None:
    match = lexical_match("Mozgov", "На площадку выходит Мозгов")

    assert match.score >= 0.82
    assert match.strategy == "transliteration"


def test_near_phonetic_asr_spelling_is_recoverable_but_not_exact() -> None:
    match = lexical_match("Мозгов", "Москов снова получает передачу")

    assert 0.68 <= match.score < 0.95
    assert match.strategy in {"fuzzy", "phonetic"}


def test_unrelated_short_word_does_not_match() -> None:
    assert lexical_match("мяч", "матч начался").score == 0


def test_glossary_expands_aliases_in_both_directions(tmp_path) -> None:
    lexicon = SearchLexicon(tmp_path / "glossary.json")
    lexicon.replace({"трёхочковый": ["треха", "три очка"]})

    assert "три очка" in lexicon.expand("трёхочковый бросок")
    assert "трёхочковый" in lexicon.expand("треха")

