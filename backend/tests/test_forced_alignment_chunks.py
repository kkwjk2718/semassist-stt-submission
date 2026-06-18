from training.forced_alignment_chunks import AlignedWord, chunk_aligned_words, sanitize_romanized


def test_sanitize_romanized_keeps_mms_supported_symbols() -> None:
    assert sanitize_romanized("Annyeong-ha.se.yo!") == "annyeonghaseyo"
    assert sanitize_romanized("can't") == "can't"


def test_chunk_aligned_words_splits_before_max_seconds() -> None:
    words = (
        AlignedWord(text="첫", romanized="cheot", start_seconds=0.0, end_seconds=5.0, score=0.9),
        AlignedWord(text="번째", romanized="beonjjae", start_seconds=5.5, end_seconds=14.0, score=0.8),
        AlignedWord(text="둘", romanized="dul", start_seconds=31.0, end_seconds=33.0, score=0.7),
    )

    chunks = chunk_aligned_words(words, max_chunk_seconds=30.0)

    assert len(chunks) == 2
    assert chunks[0].text == "첫 번째"
    assert chunks[0].duration_seconds == 14.0
    assert chunks[1].text == "둘"
    assert chunks[1].duration_seconds == 2.0
