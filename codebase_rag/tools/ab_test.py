from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class VariantConfig:
    name: str
    provider: str
    orchestrator_model: str | None
    cypher_model: str | None


@dataclass
class RunResult:
    query: str
    variant: str
    status: str
    answer: str | None
    plan: Any | None
    citations: Any | None
    latency_ms: float
    repo: str | None
    error: str | None


def post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:  # nosec - localhost expected
        body = resp.read().decode("utf-8")
        return json.loads(body)


def call_ask(
    base_url: str,
    repo_path: str,
    question: str,
    variant: VariantConfig,
) -> RunResult:
    t0 = time.perf_counter()
    error: str | None = None
    answer: str | None = None
    plan: Any | None = None
    citations: Any | None = None
    status: str = "error"
    repo_returned: str | None = None
    try:
        payload: dict[str, Any] = {
            "repo_path": repo_path,
            "question": question,
            "provider": variant.provider,
        }
        if variant.orchestrator_model:
            payload["orchestrator_model"] = variant.orchestrator_model
        if variant.cypher_model:
            payload["cypher_model"] = variant.cypher_model

        res = post_json(base_url.rstrip("/") + "/ask", payload)
        status = str(res.get("status") or "error")
        answer = res.get("answer")
        plan = res.get("plan")
        citations = res.get("citations")
        repo_returned = res.get("repo")
    except HTTPError as he:  # type: ignore
        try:
            body = he.read().decode("utf-8")
        except Exception:
            body = str(he)
        error = f"HTTPError {he.code}: {body}"
    except URLError as ue:  # type: ignore
        error = f"URLError: {ue.reason}"
    except Exception as e:
        error = str(e)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return RunResult(
        query=question,
        variant=variant.name,
        status=status,
        answer=answer,
        plan=plan,
        citations=citations,
        latency_ms=latency_ms,
        repo=repo_returned,
        error=error,
    )


def load_queries(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    # Accept either JSONL or newline-separated text
    queries: list[str] = []
    if path.suffix.lower() in {".jsonl", ".json"}:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and isinstance(obj.get("q"), str):
                    queries.append(obj["q"].strip())
                elif isinstance(obj, str):
                    queries.append(obj.strip())
            except Exception:
                continue
    else:
        queries = [q.strip() for q in text.splitlines() if q.strip()]
    # De-duplicate while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq


def write_jsonl(path: Path, rows: list[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows: list[RunResult]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "query",
            "variant",
            "status",
            "latency_ms",
            "answer_len",
            "has_plan",
            "has_citations",
            "error",
        ])
        for r in rows:
            w.writerow([
                r.query,
                r.variant,
                r.status,
                f"{r.latency_ms:.1f}",
                len(r.answer or ""),
                1 if r.plan is not None else 0,
                1 if r.citations is not None else 0,
                r.error or "",
            ])


def main() -> None:
    ap = argparse.ArgumentParser(description="Run A/B testing over random queries against /ask")
    ap.add_argument("--repo", required=True, help="Repository path ingested by the server")
    ap.add_argument("--server", default="http://127.0.0.1:8000", help="Base URL for the API server")
    ap.add_argument("--queries", required=True, help="Path to queries file (txt or jsonl)")
    ap.add_argument("--sample", type=int, default=100, help="Number of queries to sample")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    # Variant A
    ap.add_argument("--nameA", default="A", help="Label for variant A")
    ap.add_argument("--providerA", default="local", choices=["local", "gemini"], help="Provider for A")
    ap.add_argument("--orchA", default=None, help="Orchestrator model for A")
    ap.add_argument("--cypherA", default=None, help="Cypher model for A")

    # Variant B
    ap.add_argument("--nameB", default="B", help="Label for variant B")
    ap.add_argument("--providerB", default="local", choices=["local", "gemini"], help="Provider for B")
    ap.add_argument("--orchB", default=None, help="Orchestrator model for B")
    ap.add_argument("--cypherB", default=None, help="Cypher model for B")

    ap.add_argument("--out", default=".tmp/ab_results.jsonl", help="Path to write JSONL results")
    ap.add_argument("--summary", default=".tmp/ab_summary.csv", help="Path to write CSV summary")

    args = ap.parse_args()
    base_url = args.server
    repo_path = args.repo
    queries_path = Path(args.queries)
    results_path = Path(args.out)
    summary_path = Path(args.summary)

    all_queries = load_queries(queries_path)
    if not all_queries:
        raise SystemExit("No queries found")

    rng = random.Random(args.seed)
    sample = all_queries.copy()
    rng.shuffle(sample)
    sample = sample[: max(1, int(args.sample))]

    varA = VariantConfig(args.nameA, args.providerA, args.orchA, args.cypherA)
    varB = VariantConfig(args.nameB, args.providerB, args.orchB, args.cypherB)

    runs: list[RunResult] = []
    # Interleave A and B per-query to minimize drift
    for q in sample:
        runs.append(call_ask(base_url, repo_path, q, varA))
        runs.append(call_ask(base_url, repo_path, q, varB))

    write_jsonl(results_path, runs)
    write_summary_csv(summary_path, runs)

    # Simple stdout summary
    total = len(runs)
    okA = sum(1 for r in runs if r.variant == varA.name and r.status == "ok")
    okB = sum(1 for r in runs if r.variant == varB.name and r.status == "ok")
    latA = [r.latency_ms for r in runs if r.variant == varA.name]
    latB = [r.latency_ms for r in runs if r.variant == varB.name]
    avgA = sum(latA) / len(latA) if latA else 0.0
    avgB = sum(latB) / len(latB) if latB else 0.0
    print(
        json.dumps(
            {
                "queries": len(sample),
                "runs": total,
                "ok_A": okA,
                "ok_B": okB,
                "avg_latency_ms_A": round(avgA, 1),
                "avg_latency_ms_B": round(avgB, 1),
                "results_path": str(results_path),
                "summary_path": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


