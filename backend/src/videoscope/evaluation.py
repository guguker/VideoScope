from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Protocol


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    query: str
    video_id: str
    start: float
    end: float
    mode: str = "all"
    label_source: str = "gold"
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    query: str
    relevant_rank: int | None
    temporal_iou: float
    latency_ms: float
    result_count: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class VariantMetrics:
    name: str
    case_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    mean_temporal_iou: float
    mean_latency_ms: float
    cases: list[CaseEvaluation]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    generated_at: str
    variants: list[VariantMetrics]


class SearchRunner(Protocol):
    def search(
        self,
        query: str,
        *,
        video_ids: list[str] | None = None,
        limit: int = 20,
        use_lighthouse: bool = True,
        mode: str = "all",
    ) -> list[object]: ...


VARIANTS: dict[str, tuple[str, bool]] = {
    "auto": ("all", False),
    "auto_lighthouse": ("all", True),
    "speech": ("speech", False),
    "visual": ("visual", False),
    "visual_lighthouse": ("visual", True),
    "ocr": ("ocr", False),
}


def temporal_iou(first_start: float, first_end: float, second_start: float, second_end: float) -> float:
    intersection = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    union = max(first_end, second_end) - min(first_start, second_start)
    return intersection / union if union > 0 else 0.0


class EvaluationStore:
    def __init__(self, cases_path: Path, report_path: Path) -> None:
        self.cases_path = Path(cases_path)
        self.report_path = Path(report_path)

    @staticmethod
    def _write(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def read_cases(self) -> list[EvaluationCase]:
        if not self.cases_path.is_file():
            return []
        try:
            payload = json.loads(self.cases_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        raw_cases = payload.get("cases", []) if isinstance(payload, dict) else []
        output: list[EvaluationCase] = []
        for raw in raw_cases:
            if not isinstance(raw, dict):
                continue
            try:
                output.append(EvaluationCase(**raw))
            except (TypeError, ValueError):
                continue
        return output

    def replace_cases(self, cases: list[EvaluationCase]) -> None:
        seen: set[str] = set()
        for case in cases:
            if not case.id.strip() or not case.query.strip() or not case.video_id.strip():
                raise ValueError("evaluation case fields must not be empty")
            if case.end <= case.start or case.start < 0:
                raise ValueError("evaluation case interval must be positive and ordered")
            if case.id in seen:
                raise ValueError("evaluation case ids must be unique")
            seen.add(case.id)
        self._write(self.cases_path, {"version": 1, "cases": [asdict(case) for case in cases]})

    def write_report(self, report: EvaluationReport) -> None:
        self._write(self.report_path, asdict(report))

    def read_report(self) -> dict[str, object] | None:
        if not self.report_path.is_file():
            return None
        try:
            payload = json.loads(self.report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None


class EvaluationService:
    def __init__(self, search: SearchRunner, store: EvaluationStore) -> None:
        self.search = search
        self.store = store

    def run(self, variants: list[str] | None = None) -> EvaluationReport:
        selected = variants or ["auto"]
        unknown = set(selected) - VARIANTS.keys()
        if unknown:
            raise ValueError(f"unsupported evaluation variants: {sorted(unknown)}")
        cases = self.store.read_cases()
        report = EvaluationReport(
            generated_at=datetime.now(UTC).isoformat(),
            variants=[self._run_variant(name, cases) for name in selected],
        )
        self.store.write_report(report)
        return report

    def _run_variant(self, name: str, cases: list[EvaluationCase]) -> VariantMetrics:
        mode, use_lighthouse = VARIANTS[name]
        evaluations: list[CaseEvaluation] = []
        for case in cases:
            started = perf_counter()
            error = None
            try:
                results = self.search.search(
                    case.query,
                    video_ids=[case.video_id],
                    limit=20,
                    use_lighthouse=use_lighthouse,
                    mode=mode,
                )
            except Exception as caught:
                results = []
                error = str(caught)
            latency_ms = (perf_counter() - started) * 1000
            relevant_rank = None
            overlap = 0.0
            for rank, result in enumerate(results, start=1):
                if str(getattr(result, "video_id", "")) != case.video_id:
                    continue
                current_iou = temporal_iou(
                    case.start,
                    case.end,
                    float(getattr(result, "start", 0.0)),
                    float(getattr(result, "end", 0.0)),
                )
                if current_iou > 0:
                    relevant_rank = rank
                    overlap = current_iou
                    break
            evaluations.append(
                CaseEvaluation(
                    case_id=case.id,
                    query=case.query,
                    relevant_rank=relevant_rank,
                    temporal_iou=overlap,
                    latency_ms=latency_ms,
                    result_count=len(results),
                    error=error,
                )
            )

        count = len(evaluations)
        divisor = count or 1
        return VariantMetrics(
            name=name,
            case_count=count,
            recall_at_1=sum(item.relevant_rank is not None and item.relevant_rank <= 1 for item in evaluations) / divisor,
            recall_at_3=sum(item.relevant_rank is not None and item.relevant_rank <= 3 for item in evaluations) / divisor,
            recall_at_5=sum(item.relevant_rank is not None and item.relevant_rank <= 5 for item in evaluations) / divisor,
            mrr=sum(1 / item.relevant_rank for item in evaluations if item.relevant_rank) / divisor,
            mean_temporal_iou=mean(item.temporal_iou for item in evaluations) if evaluations else 0.0,
            mean_latency_ms=mean(item.latency_ms for item in evaluations) if evaluations else 0.0,
            cases=evaluations,
        )
