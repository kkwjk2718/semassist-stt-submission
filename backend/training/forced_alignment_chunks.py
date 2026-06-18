from __future__ import annotations

import re
from dataclasses import dataclass

MMS_TOKEN_RE = re.compile(r"[^a-z']+")


@dataclass(frozen=True, slots=True)
class AlignedWord:
    text: str
    romanized: str
    start_seconds: float
    end_seconds: float
    score: float


@dataclass(frozen=True, slots=True)
class AlignedChunk:
    text: str
    start_seconds: float
    end_seconds: float
    mean_score: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def sanitize_romanized(value: str) -> str:
    return MMS_TOKEN_RE.sub("", value.casefold())


def chunk_aligned_words(words: tuple[AlignedWord, ...], max_chunk_seconds: float) -> tuple[AlignedChunk, ...]:
    if max_chunk_seconds <= 0:
        raise RuntimeError(f"max_chunk_seconds must be positive: {max_chunk_seconds}")
    chunks: list[AlignedChunk] = []
    current: list[AlignedWord] = []
    for word in words:
        if current and word.end_seconds - current[0].start_seconds > max_chunk_seconds:
            chunks.append(_build_chunk(tuple(current)))
            current = []
        current.append(word)
    if current:
        chunks.append(_build_chunk(tuple(current)))
    return tuple(chunks)


def _build_chunk(words: tuple[AlignedWord, ...]) -> AlignedChunk:
    if not words:
        raise RuntimeError("cannot build an empty aligned chunk")
    return AlignedChunk(
        text=" ".join(word.text for word in words),
        start_seconds=words[0].start_seconds,
        end_seconds=words[-1].end_seconds,
        mean_score=sum(word.score for word in words) / len(words),
    )
