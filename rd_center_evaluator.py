#!/usr/bin/env python3
"""
R&D Center Feasibility Evaluation Module

Core logic: PDF text extraction, LLM-based evaluation against Law No. 5746 criteria,
and report generation. Can be used standalone (CLI) or imported by server.py.

Usage (standalone):
    python rd_center_evaluator.py sample.pdf --api-key sk-... --output report.md

Requirements:
    pip install openai pymupdf
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
from openai import OpenAI


# ============================================================================
# R&D Center Criteria (Law No. 5746 + Regulation + CB Decision 6652)
# ============================================================================

RD_CENTER_CRITERIA = {
    "C1": {
        "name": "R&D Personnel Count",
        "description": "Minimum 15 FTE R&D personnel (30 for automotive/transport manufacturing per NACE codes 29-30).",
        "category": "mandatory",
    },
    "C2": {
        "name": "Domestic R&D Activities",
        "description": "All R&D activities covered under the Law must be conducted within Turkey.",
        "category": "mandatory",
    },
    "C3": {
        "name": "R&D Management Capability",
        "description": "The applicant must demonstrate sufficient R&D management capacity including technological assets, HR, IP rights, and project/knowledge resource management.",
        "category": "mandatory",
    },
    "C4": {
        "name": "Physical Access Control",
        "description": "The R&D center must have mechanisms for physical verification that R&D and support personnel are working on-site.",
        "category": "mandatory",
    },
    "C5": {
        "name": "Defined R&D Projects",
        "description": "The center must have R&D/innovation programs and projects with defined scope, duration, budget, and personnel requirements.",
        "category": "mandatory",
    },
    "C6": {
        "name": "Separate Organizational Unit",
        "description": "The R&D center must be organized as a separate unit within the company and located in a single campus or physical space.",
        "category": "mandatory",
    },
    "P1": {
        "name": "R&D Expenditure / Revenue Ratio",
        "description": "Ratio of R&D or design expenditure to total revenue.",
        "category": "performance",
    },
    "P2": {
        "name": "Registered Patents",
        "description": "Number of registered national or international patents.",
        "category": "performance",
    },
    "P3": {
        "name": "Postgraduate Researcher Ratio",
        "description": "Ratio of postgraduate-degree researchers to total R&D personnel.",
        "category": "performance",
    },
    "P4": {
        "name": "Total Researcher Ratio",
        "description": "Ratio of total researchers to total R&D personnel count.",
        "category": "performance",
    },
    "P5": {
        "name": "New Product Revenue Ratio",
        "description": "Ratio of revenue from new products resulting from R&D to total revenue.",
        "category": "performance",
    },
    "P6": {
        "name": "University-Industry Collaboration",
        "description": "Collaboration with universities or research institutions (joint projects, thesis supervision, publications).",
        "category": "performance",
    },
    "P7": {
        "name": "International Project Support",
        "description": "Number of internationally supported R&D projects.",
        "category": "performance",
    },
}


# ============================================================================
# PDF Processing
# ============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts full text from a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    full_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        full_text.append(f"--- Page {page_num + 1} ---\n{text}")
    doc.close()

    combined = "\n\n".join(full_text)
    if not combined.strip():
        raise ValueError("No text could be extracted from the PDF. OCR may be required.")
    return combined


def get_pdf_metadata(pdf_path: str) -> dict:
    """Returns metadata of the PDF file."""
    doc = fitz.open(pdf_path)
    metadata = {
        "page_count": len(doc),
        "file_size_kb": round(os.path.getsize(pdf_path) / 1024, 1),
        "title": doc.metadata.get("title", "") or "N/A",
        "author": doc.metadata.get("author", "") or "N/A",
    }
    doc.close()
    return metadata


# ============================================================================
# LLM Evaluation
# ============================================================================

def build_system_prompt() -> str:
    """Builds the system prompt with all R&D center criteria."""
    mandatory = []
    performance = []
    for cid, c in RD_CENTER_CRITERIA.items():
        line = f"- [{cid}] {c['name']}: {c['description']}"
        if c["category"] == "mandatory":
            mandatory.append(line)
        else:
            performance.append(line)

    return f"""You are an expert R&D Center Feasibility Evaluator for the Turkish Ministry of Industry and Technology.
You evaluate company documents under Law No. 5746 and the associated Implementation and Audit Regulation.

MANDATORY APPLICATION PREREQUISITES (all must be met):
{chr(10).join(mandatory)}

PERFORMANCE INDICATORS (CB Decision 6652, replacing BKK 2016/9092):
{chr(10).join(performance)}
A minimum 20% year-over-year improvement in any one performance indicator qualifies for additional R&D tax deduction (50% of increase, valid until 31/12/2028).

SECTOR-SPECIFIC EXCEPTION:
NACE codes 29 (motor vehicles) and 30 (other transport) require minimum 30 FTE instead of 15.

YOUR TASK:
1. Analyze the uploaded document(s) thoroughly.
2. For each mandatory criterion (C1-C6), determine: MET / PARTIALLY MET / NOT MET / INSUFFICIENT DATA.
3. For each performance indicator (P1-P7), assign a score from 0-5 and note findings.
4. Provide an overall feasibility score (0-100) and eligibility status.
5. List critical gaps and a concrete action plan.

RESPOND STRICTLY IN THIS JSON FORMAT:
{{
  "company_name": "...",
  "sector": "...",
  "evaluation_date": "{datetime.now().strftime('%Y-%m-%d')}",
  "mandatory_criteria": [
    {{
      "id": "C1",
      "name": "...",
      "status": "MET | PARTIALLY MET | NOT MET | INSUFFICIENT DATA",
      "findings": "...",
      "evidence": "...",
      "recommendations": ["..."]
    }}
  ],
  "performance_indicators": [
    {{
      "id": "P1",
      "name": "...",
      "score": 0-5,
      "findings": "...",
      "recommendations": ["..."]
    }}
  ],
  "overall_score": 0-100,
  "eligibility_status": "ELIGIBLE | CONDITIONALLY ELIGIBLE | NOT ELIGIBLE",
  "overall_assessment": "...",
  "critical_gaps": ["..."],
  "action_plan": ["..."],
  "estimated_incentives": {{
    "income_tax_exemption": "...",
    "social_security_support": "...",
    "stamp_duty_exemption": "...",
    "rd_tax_deduction": "..."
  }}
}}

Return ONLY valid JSON, nothing else."""


def build_user_prompt(pdf_text: str) -> str:
    """Builds the user prompt with document content."""
    max_chars = 60000
    if len(pdf_text) > max_chars:
        pdf_text = pdf_text[:max_chars] + "\n\n[... Document truncated ...]"

    return f"""Evaluate the following company document for R&D Center eligibility under Turkish Law No. 5746:

--- DOCUMENT START ---
{pdf_text}
--- DOCUMENT END ---

Analyze all mandatory criteria (C1-C6) and performance indicators (P1-P7). Respond in JSON."""


def run_evaluation(api_key: str, pdf_text: str, model: str = "gpt-4o-mini") -> dict:
    """Runs LLM evaluation and returns structured result."""
    client = OpenAI(api_key=api_key)
    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(pdf_text)},
        ],
        temperature=0.2,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    elapsed = time.time() - start_time
    content = response.choices[0].message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"raw_response": content, "parse_error": True}

    result["_meta"] = {
        "model": model,
        "elapsed_seconds": round(elapsed, 2),
        "token_usage": usage,
        "timestamp": datetime.now().isoformat(),
    }
    return result


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(evaluation: dict, pdf_metadata: dict) -> str:
    """Generates a Markdown report from evaluation result."""
    lines = []
    lines.append("# R&D Center Feasibility Evaluation Report\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Company:** {evaluation.get('company_name', 'N/A')}")
    lines.append(f"**Sector:** {evaluation.get('sector', 'N/A')}\n")

    lines.append("## Document Information\n")
    lines.append(f"| Property | Value |\n|----------|-------|\n| Pages | {pdf_metadata.get('page_count', '-')} |\n| Size | {pdf_metadata.get('file_size_kb', '-')} KB |\n")

    # Mandatory criteria
    mandatory = evaluation.get("mandatory_criteria", [])
    if mandatory:
        lines.append("## Mandatory Criteria (C1-C6)\n")
        for c in mandatory:
            status = c.get("status", "?")
            icon = {"MET": "✅", "PARTIALLY MET": "⚠️", "NOT MET": "❌"}.get(status, "❓")
            lines.append(f"### {icon} {c.get('id', '?')} — {c.get('name', '?')} [{status}]\n")
            lines.append(f"**Findings:** {c.get('findings', '-')}\n")
            if c.get("evidence"):
                lines.append(f"**Evidence:** {c.get('evidence', '-')}\n")
            recs = c.get("recommendations", [])
            if recs:
                lines.append("**Recommendations:**")
                for r in recs:
                    lines.append(f"  - {r}")
                lines.append("")

    # Performance indicators
    perf = evaluation.get("performance_indicators", [])
    if perf:
        lines.append("## Performance Indicators (P1-P7)\n")
        for p in perf:
            score = p.get("score", "?")
            bar = "█" * int(score) + "░" * (5 - int(score)) if isinstance(score, int) else ""
            lines.append(f"### {p.get('id', '?')} — {p.get('name', '?')} [{score}/5] {bar}\n")
            lines.append(f"**Findings:** {p.get('findings', '-')}\n")
            recs = p.get("recommendations", [])
            if recs:
                lines.append("**Recommendations:**")
                for r in recs:
                    lines.append(f"  - {r}")
                lines.append("")

    # Overall
    lines.append("## Overall Assessment\n")
    lines.append(f"**Score:** {evaluation.get('overall_score', '?')}/100")
    lines.append(f"**Eligibility:** {evaluation.get('eligibility_status', '?')}\n")
    lines.append(f"{evaluation.get('overall_assessment', '')}\n")

    gaps = evaluation.get("critical_gaps", [])
    if gaps:
        lines.append("## Critical Gaps\n")
        for i, g in enumerate(gaps, 1):
            lines.append(f"{i}. {g}")
        lines.append("")

    actions = evaluation.get("action_plan", [])
    if actions:
        lines.append("## Action Plan\n")
        for i, a in enumerate(actions, 1):
            lines.append(f"{i}. {a}")
        lines.append("")

    incentives = evaluation.get("estimated_incentives", {})
    if incentives:
        lines.append("## Estimated Incentives\n")
        lines.append("| Incentive | Detail |\n|-----------|--------|")
        for k, v in incentives.items():
            lines.append(f"| {k.replace('_', ' ').title()} | {v} |")
        lines.append("")

    meta = evaluation.get("_meta", {})
    if meta:
        lines.append("---\n")
        lines.append(f"*Model: {meta.get('model', '?')} | Time: {meta.get('elapsed_seconds', '?')}s | Tokens: {meta.get('token_usage', {}).get('total_tokens', '?')}*")

    return "\n".join(lines)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="R&D Center Feasibility Evaluator")
    parser.add_argument("pdf", help="Path to the PDF file to evaluate")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--output", "-o", default=None, help="Output report path")
    parser.add_argument("--json", action="store_true", help="Also save raw JSON")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: API key required. Use --api-key or set OPENAI_API_KEY.")
        sys.exit(1)

    if args.output is None:
        args.output = f"{Path(args.pdf).stem}_report.md"

    print("=" * 55)
    print("  R&D Center Feasibility Evaluation System")
    print("=" * 55)

    # 1. Extract PDF
    print("\n[1/3] Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(args.pdf)
    pdf_meta = get_pdf_metadata(args.pdf)
    print(f"  ✓ {pdf_meta['page_count']} pages, {pdf_meta['file_size_kb']} KB")

    # 2. Evaluate
    print("\n[2/3] Running AI evaluation...")
    result = run_evaluation(args.api_key, pdf_text, args.model)
    print(f"  ✓ Done in {result['_meta']['elapsed_seconds']}s ({result['_meta']['token_usage']['total_tokens']} tokens)")

    # 3. Generate report
    print("\n[3/3] Generating report...")
    report = generate_report(result, pdf_meta)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  ✓ Report saved: {args.output}")

    if args.json:
        json_path = args.output.replace(".md", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  ✓ JSON saved: {json_path}")

    # Summary
    print(f"\n{'=' * 55}")
    elig = result.get("eligibility_status", "?")
    score = result.get("overall_score", "?")
    print(f"  RESULT: {elig} (Score: {score}/100)")
    print("=" * 55)


if __name__ == "__main__":
    main()