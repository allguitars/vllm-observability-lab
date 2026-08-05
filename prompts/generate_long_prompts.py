#!/usr/bin/env python3

"""Generate three deterministic long prompts for prefix-cache experiments."""

from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = (
    ROOT / "tokenizers" / "gemma-3-27b-it" / "tokenizer.json"
)
OUTPUT_DIR = ROOT / "prompts"
MIN_CHAT_TOKENS = 7600
MAX_CHAT_TOKENS = 7800


DOMAINS = {
    "prefix_a_distributed_systems.txt": {
        "marker": "PREFIX_TEST_A_20260801_DISTRIBUTED_SYSTEMS",
        "title": "Reliability Review for a Distributed Inference Platform",
        "subjects": [
            "the admission controller",
            "the replicated metadata service",
            "the model-serving scheduler",
            "the cross-region event stream",
            "the object-storage gateway",
            "the feature retrieval layer",
            "the GPU worker pool",
            "the service discovery plane",
        ],
        "actions": [
            "applies bounded backpressure before accepting additional work",
            "records an idempotency key before publishing a state transition",
            "separates readiness from liveness during partial dependency failures",
            "uses monotonic sequence numbers to reject stale updates",
            "limits retries with exponential delay and randomized jitter",
            "moves oversized batches into a lower-priority scheduling class",
            "verifies ownership through a renewable lease",
            "preserves request lineage across asynchronous boundaries",
        ],
        "evidence": [
            "queue depth, tail latency, and cancellation rate",
            "lease age, replication lag, and conflict frequency",
            "time to first token, batch occupancy, and preemption count",
            "checkpoint duration, replay distance, and recovery time",
            "cache hit ratio, transfer bytes, and eviction frequency",
            "error-budget consumption and dependency saturation",
        ],
        "risks": [
            "a retry storm after a regional network interruption",
            "silent duplication when a client loses the final acknowledgement",
            "priority inversion between interactive and background workloads",
            "stale routing information during a rolling deployment",
            "resource fragmentation caused by incompatible request shapes",
            "correlated failure across services sharing the same control plane",
        ],
    },
    "prefix_b_financial_risk.txt": {
        "marker": "PREFIX_TEST_B_20260801_FINANCIAL_RISK",
        "title": "Independent Risk Review for a Multi-Asset Portfolio",
        "subjects": [
            "the market-risk desk",
            "the collateral management team",
            "the treasury function",
            "the counterparty credit group",
            "the valuation control unit",
            "the liquidity risk committee",
            "the derivatives operations team",
            "the portfolio construction group",
        ],
        "actions": [
            "reconciles position data before calculating daily exposure",
            "applies conservative haircuts to assets with limited market depth",
            "separates model uncertainty from observable price volatility",
            "tests concentrated positions under discontinuous price moves",
            "compares internal valuations with independent market evidence",
            "limits wrong-way risk through counterparty-specific thresholds",
            "maps contractual cash flows into maturity and currency buckets",
            "documents overrides with an accountable approval trail",
        ],
        "evidence": [
            "expected shortfall, stress loss, and concentration measures",
            "variation margin, collateral disputes, and settlement failures",
            "funding gaps, liquid-asset coverage, and rollover assumptions",
            "probability of default, exposure at default, and recovery estimates",
            "valuation reserves, price dispersion, and stale quote frequency",
            "scenario loss, hedge effectiveness, and basis risk",
        ],
        "risks": [
            "a rapid volatility increase combined with widening bid-ask spreads",
            "a collateral shortfall during an intraday margin call",
            "a crowded exit from a position with limited secondary demand",
            "an adverse correlation shift that weakens an assumed hedge",
            "a counterparty downgrade coinciding with rising replacement cost",
            "a funding disruption across several currencies and maturities",
        ],
    },
    "prefix_c_biomedical_research.txt": {
        "marker": "PREFIX_TEST_C_20260801_BIOMEDICAL_RESEARCH",
        "title": "Protocol and Evidence Review for a Multicenter Clinical Study",
        "subjects": [
            "the clinical coordinating center",
            "the central laboratory",
            "the site monitoring team",
            "the independent safety committee",
            "the biostatistics group",
            "the pharmacovigilance unit",
            "the imaging core laboratory",
            "the data management team",
        ],
        "actions": [
            "verifies eligibility before treatment allocation",
            "preserves specimen identity through a documented chain of custody",
            "records protocol deviations before the database is locked",
            "reviews unblinded safety data under a predefined charter",
            "models missing outcomes using prespecified sensitivity analyses",
            "classifies adverse events with consistent medical terminology",
            "uses blinded quality checks for acquisition and interpretation",
            "maintains a traceable audit history for every corrected field",
        ],
        "evidence": [
            "recruitment rate, screen failures, and withdrawal patterns",
            "assay precision, specimen stability, and batch effects",
            "treatment adherence, visit completion, and missing observations",
            "serious adverse events, laboratory trends, and stopping boundaries",
            "effect estimates, confidence intervals, and sensitivity results",
            "inter-reader agreement, image quality, and endpoint adjudication",
        ],
        "risks": [
            "differential loss to follow-up between treatment groups",
            "measurement drift across laboratories and enrollment periods",
            "unblinding caused by recognizable treatment side effects",
            "selective reporting after an unexpected subgroup result",
            "site-level variation in supportive care and outcome assessment",
            "insufficient representation of patients with important comorbidities",
        ],
    },
}


def make_section(domain: dict[str, object], index: int) -> str:
    subjects = domain["subjects"]
    actions = domain["actions"]
    evidence = domain["evidence"]
    risks = domain["risks"]
    subject = subjects[index % len(subjects)]
    second_subject = subjects[(index * 3 + 1) % len(subjects)]
    action = actions[(index * 5 + 2) % len(actions)]
    second_action = actions[(index * 7 + 3) % len(actions)]
    measure = evidence[(index * 3 + 1) % len(evidence)]
    second_measure = evidence[(index * 5 + 2) % len(evidence)]
    risk = risks[(index * 7 + 1) % len(risks)]
    second_risk = risks[(index * 11 + 3) % len(risks)]

    return (
        f"Section {index:03d}: Control objective and observed conditions\n\n"
        f"In review cycle {index:03d}, {subject} {action}. The control is evaluated "
        f"against {measure}, because an isolated average can conceal changes in the "
        f"tail of the distribution. Reviewers should preserve timestamps, decision "
        f"inputs, and ownership records so that later analysis can distinguish an "
        f"expected operational variation from a genuine control failure.\n\n"
        f"The primary scenario is {risk}. Under that scenario, {second_subject} "
        f"{second_action}. The assessment must state which assumptions remain valid, "
        f"which dependencies have become unavailable, and which actions are safe to "
        f"repeat. A recovery that restores service but corrupts ordering, attribution, "
        f"or study evidence is not considered successful.\n\n"
        f"Evidence for this section includes {second_measure}. Compare the baseline "
        f"with a moderate disturbance and a severe but plausible disturbance. Also "
        f"test {second_risk}. Record both the immediate response and the delayed "
        f"consequence, since a locally successful intervention can transfer risk to "
        f"another component, reporting period, institution, or stakeholder group.\n"
    )


def chat_token_count(tokenizer: Tokenizer, text: str) -> int:
    # Equivalent to the repository's Gemma 3 chat template for one user turn
    # with add_generation_prompt=True.
    rendered_prompt = (
        "<bos><start_of_turn>user\n"
        f"{text.strip()}"
        "<end_of_turn>\n<start_of_turn>model\n"
    )
    return len(tokenizer.encode(rendered_prompt, add_special_tokens=False).ids)


def build_prompt(tokenizer, domain: dict[str, object]) -> tuple[str, int]:
    parts = [
        f"{domain['marker']}\n\n",
        f"# {domain['title']}\n\n",
        "Read the complete reference document below. Retain its exact sequence and "
        "terminology for a later question. Do not summarize it yet. After reading, "
        "reply with only the word READY.\n\n",
    ]

    index = 1
    while True:
        candidate = "".join(parts + [make_section(domain, index)])
        candidate_count = chat_token_count(tokenizer, candidate)
        if candidate_count > MAX_CHAT_TOKENS:
            break
        parts.append(make_section(domain, index))
        index += 1

    text = "".join(parts).rstrip() + "\n"
    count = chat_token_count(tokenizer, text)
    note_index = 1
    while count < MIN_CHAT_TOKENS:
        text += (
            f"\nSupplemental note {note_index:02d}: Preserve this observation in "
            "sequence and evaluate it with the same evidence standard described "
            "above."
        )
        count = chat_token_count(tokenizer, text)
        note_index += 1
    if not MIN_CHAT_TOKENS <= count <= MAX_CHAT_TOKENS:
        raise RuntimeError(f"Generated prompt has {count} chat tokens")
    return text, count


def main() -> None:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    OUTPUT_DIR.mkdir(exist_ok=True)

    for filename, domain in DOMAINS.items():
        text, count = build_prompt(tokenizer, domain)
        output_path = OUTPUT_DIR / filename
        output_path.write_text(text, encoding="utf-8")
        print(f"{filename}: {count} chat tokens, {len(text)} characters")


if __name__ == "__main__":
    main()
