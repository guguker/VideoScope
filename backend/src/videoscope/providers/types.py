from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TimedText:
    start: float
    end: float
    text: str
    confidence: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ObjectTag:
    label: str
    confidence: float
    metadata: dict[str, object] = field(default_factory=dict)

