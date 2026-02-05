"""
4-Stage RAG Pipeline Evaluation Script

Tests each stage independently:
  python evaluate_pipeline.py --test all           # Full pipeline
  python evaluate_pipeline.py --test ingest        # Ingest docs only
  python evaluate_pipeline.py --test chunking      # Test Case 1: Chunking + Metadata
  python evaluate_pipeline.py --test retrieval     # Test Case 2: Retrieval
  python evaluate_pipeline.py --test reranking     # Test Case 3: Reranking (Cross-Encoder)
  python evaluate_pipeline.py --test generation    # Test Case 4: Generation
"""

import argparse
import asyncio
import json
import os
import sys
import io
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings

# Paths
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "OneDrive_1_2-2-2026")
QA_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "LIC_QA_Evaluation_Results.xlsx")
SAMPLE_PDF = os.path.join(DOCS_DIR, "wholelife_plan", "102268- Jeevan Utsav Sales Brochure.pdf")

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "of", "in", "to",
    "for", "with", "on", "at", "by", "from", "as", "or", "and", "if",
    "it", "its", "this", "that", "not", "no", "but", "so", "than",
}


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_metric(label: str, value, width: int = 35):
    print(f"  {label:<{width}} {value}")


def keyword_overlap(text_a: str, text_b: str) -> float:
    """Compute keyword overlap between two texts (0-1)."""
    if not text_a or not text_b:
        return 0.0
    words_a = set(text_a.lower().split()) - STOPWORDS
    words_b = set(text_b.lower().split()) - STOPWORDS
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


def load_qa_dataset() -> List[Dict[str, str]]:
    """Load Q&A evaluation dataset from Excel."""
    import pandas as pd

    if not os.path.exists(QA_PATH):
        print(f"  Q&A dataset not found: {QA_PATH}")
        return []

    df = pd.read_excel(QA_PATH)
    questions = []
    for _, row in df.iterrows():
        q = str(row.get("Sample Questions", "")).strip()
        if q and q.lower() != "nan":
            questions.append({
                "question": q,
                "expected": str(row.get("Expected Answers", "")),
                "plan": str(row.get("Plan", "Unknown")),
            })
    return questions


# ─────────────────────────────────────────────────────────────────
# Stage 0: Ingestion
# ─────────────────────────────────────────────────────────────────

async def test_ingestion(limit: int = 0, use_llm_metadata: bool = False):
    """Ingest PDFs from docs folder into Qdrant."""
    from app.core.database import init_qdrant, recreate_collection
    from app.services.ingestion.service import IngestionService

    print_header("STAGE 0: DOCUMENT INGESTION")

    # Find all PDFs (deduplicate by filename)
    seen = set()
    pdf_files = []
    for root, _, files in os.walk(DOCS_DIR):
        for f in files:
            if f.lower().endswith(".pdf") and f not in seen:
                seen.add(f)
                pdf_files.append(os.path.join(root, f))

    if limit > 0:
        pdf_files = pdf_files[:limit]
        print(f"  Limited to {limit} PDFs (out of {len(seen)} found)")
    print(f"  Found {len(pdf_files)} unique PDFs in {DOCS_DIR}")

    # Recreate collection for clean evaluation
    print("  Recreating Qdrant collection...")
    recreate_collection()

    # Ingest
    service = IngestionService(use_llm_metadata=use_llm_metadata)
    total_chunks = 0
    results = []

    for i, pdf_path in enumerate(pdf_files, 1):
        fname = os.path.basename(pdf_path)
        print(f"  [{i}/{len(pdf_files)}] {fname[:50]:<50} ", end="", flush=True)
        start = time.time()
        try:
            result = await service.process_local_file(pdf_path, verbose=True)
            elapsed = time.time() - start
            chunks = result.get("chunks", 0)
            total_chunks += chunks
            results.append({"file": fname, "chunks": chunks, "time": elapsed, "status": "OK"})
            print(f" OK ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - start
            results.append({"file": fname, "chunks": 0, "time": elapsed, "status": f"FAIL: {e}"})
            print(f"[FAIL] {str(e)[:50]}")

    # Rebuild BM25 index after ingestion
    from app.core.database import rebuild_bm25_index
    print("\n  Rebuilding BM25 index...")
    try:
        rebuild_bm25_index()
        print("  BM25 index rebuilt.")
    except Exception as e:
        print(f"  [WARN] BM25 rebuild failed: {e}")

    print(f"\n  --- Ingestion Summary ---")
    print_metric("Total documents:", len(pdf_files))
    print_metric("Successful:", sum(1 for r in results if r["status"] == "OK"))
    print_metric("Failed:", sum(1 for r in results if r["status"] != "OK"))
    print_metric("Total chunks:", total_chunks)
    print_metric("Avg chunks/doc:", f"{total_chunks / max(len(pdf_files), 1):.1f}")

    return results


# ─────────────────────────────────────────────────────────────────
# Test Case 1: Chunking + 3-Layer Metadata
# ─────────────────────────────────────────────────────────────────

async def test_chunking_and_metadata():
    """Test chunk quality and metadata integration."""
    from app.services.ingestion.parsers import DocumentParserFactory
    from app.services.ingestion.chunkers import StructuredChunker
    from app.services.ingestion.metadata import (
        enrich_chunk_metadata,
        extract_structured_fields,
        MetadataExtractor,
        get_section_display_name,
    )

    print_header("TEST 1: CHUNKING + METADATA")

    if not os.path.exists(SAMPLE_PDF):
        print(f"  Sample PDF not found: {SAMPLE_PDF}")
        return {}

    # 1. Parse (structured)
    print(f"  Parsing: {os.path.basename(SAMPLE_PDF)}")
    parser = DocumentParserFactory.get_parser(SAMPLE_PDF)
    parsed_doc = parser.parse_structured(SAMPLE_PDF)
    text = parsed_doc.full_text
    print_metric("Extracted text length:", f"{len(text)} chars")
    print_metric("Headings detected:", len(parsed_doc.headings))
    print_metric("Tables detected:", len(parsed_doc.tables))

    # 2. Chunk (structured)
    print("\n  Running StructuredChunker...")
    chunker = StructuredChunker()
    chunks = chunker.chunk_structured(parsed_doc)

    sizes = [len(c.text) for c in chunks]
    print_metric("Chunks generated:", len(chunks))
    print_metric("Avg chunk size:", f"{sum(sizes) // len(sizes)} chars")
    print_metric("Min / Max chunk size:", f"{min(sizes)} / {max(sizes)} chars")

    # Size distribution
    print("\n  Size distribution:")
    buckets = [(0, 100), (100, 400), (400, 800), (800, 1200), (1200, 2000), (2000, 5000)]
    for lo, hi in buckets:
        count = sum(1 for s in sizes if lo <= s < hi)
        pct = 100 * count / len(sizes) if sizes else 0
        print(f"    {lo:>5}-{hi:<5}: {count:>4} ({pct:5.1f}%)")

    # Section distribution
    section_counts = defaultdict(int)
    for c in chunks:
        section_counts[c.section_type] += 1
    non_general = sum(1 for c in chunks if c.section_type != "general")
    print_metric("\n  Chunks with section != general:", f"{non_general}/{len(chunks)} ({100*non_general//len(chunks)}%)")

    print("\n  Section distribution:")
    for sec, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"    {get_section_display_name(sec):<30} {count}")

    # 3. Layer 1: Pattern-based enrichment
    print("\n  --- Layer 1: Pattern-based extraction ---")
    layer1_tag_count = 0
    layer1_fields_found = set()
    for c in chunks:
        meta = enrich_chunk_metadata(c.text, c.section_type, c.content_type)
        if meta.get("chunk_tags"):
            layer1_tag_count += 1
        for field_name in ["plan_number", "uin", "age_ranges", "monetary_amounts", "benefit_types"]:
            if meta.get(field_name):
                layer1_fields_found.add(field_name)

    print_metric("Chunks with tags:", f"{layer1_tag_count}/{len(chunks)} ({100*layer1_tag_count//len(chunks)}%)")
    print_metric("Fields found:", ", ".join(layer1_fields_found) if layer1_fields_found else "(none)")

    # 4. Layer 3: LLM-based extraction
    print("\n  --- Layer 3: LLM-based extraction ---")
    layer3_success = False
    layer3_result = {}
    try:
        extractor = MetadataExtractor()
        tables_preview = "\n\n".join(t.markdown for t in parsed_doc.tables if t.markdown)
        layer3_result = extractor.extract_metadata(
            text, os.path.basename(SAMPLE_PDF),
            headings=parsed_doc.headings,
            tables_preview=tables_preview or None,
        )
        layer3_success = bool(layer3_result.get("plan_name") or layer3_result.get("plan_type"))
        print_metric("LLM extraction:", "SUCCESS" if layer3_success else "PARTIAL")
        for k, v in layer3_result.items():
            if v and k != "keywords":
                print(f"    {k:<25} {str(v)[:60]}")
        if layer3_result.get("keywords"):
            print(f"    {'keywords':<25} {layer3_result['keywords'][:5]}")
    except Exception as e:
        print_metric("LLM extraction:", f"FAILED ({e})")

    # 5. Sample enriched chunks
    print("\n  --- Sample Enriched Chunks (first 3) ---")
    for i, c in enumerate(chunks[:3]):
        meta = enrich_chunk_metadata(c.text, c.section_type, c.content_type)
        print(f"\n  Chunk {i+1}: [{c.section_type}] [{c.content_type}] (page {c.page_number})")
        print(f"    Heading: {c.heading}")
        print(f"    Tags:    {meta.get('chunk_tags', [])[:5]}")
        print(f"    Hints:   {meta.get('entity_hints', [])}")
        print(f"    Text:    {c.text[:150].replace(chr(10), ' ')}...")

    metrics = {
        "chunk_count": len(chunks),
        "avg_size": sum(sizes) // len(sizes),
        "pct_with_section": round(100 * non_general / len(chunks), 1),
        "pct_with_tags": round(100 * layer1_tag_count / len(chunks), 1),
        "layer1_fields": list(layer1_fields_found),
        "layer3_success": layer3_success,
    }

    print(f"\n  TEST 1 RESULT: {'PASS' if layer1_tag_count > 0 else 'PARTIAL'}")
    return metrics


# ─────────────────────────────────────────────────────────────────
# Test Case 2: Retrieval Quality
# ─────────────────────────────────────────────────────────────────

# Known plan names for extracting from question text
_PLAN_NAMES = [
    "jeevan utsav", "jeevan umang", "jeevan akshay", "jeevan shanti",
    "bima shree", "bima ratna", "bima kavach", "bima jyoti",
    "amritbaal", "index plus", "nivesh plus", "new pension plus",
    "smart pension", "yuva credit", "single premium endowment",
    "new endowment", "new money back", "saral pension", "saral jeevan",
    "digi term", "digi credit",
]

# Category-level questions (no specific plan) — match by document type
_CATEGORY_KEYWORDS = {
    "endowment": "endowment",
    "money back": "moneyback",
    "moneyback": "moneyback",
    "whole life": "whole_life",
    "term assurance": "term_insurance",
    "term insurance": "term_insurance",
    "unit linked": "unit",  # matches "unit" in filenames
    "pension": "pension",
    "claim": "claim",
    "grievance": "grievance",
}


def _extract_plan_from_question(question: str) -> str:
    """Extract specific plan name from question text."""
    q_lower = question.lower()
    for plan in _PLAN_NAMES:
        if plan in q_lower:
            return plan
    return ""


def _check_chunk_relevant(doc: dict, expected: str, plan: str, question: str = "") -> bool:
    """
    Check if a retrieved chunk is relevant.

    Strategy:
    1. Extract the target plan name from the question text itself (not the Plan column).
    2. If a specific plan is identified, check that the chunk's source file or
       plan_name payload field matches that plan (source document match).
    3. For category-level questions ("What types of endowment plans..."),
       check the chunk comes from the right category document.
    4. Keyword overlap is used only as a secondary signal for generic questions
       where no plan/category can be identified.
    """
    text = doc.get("text", "").lower()
    source = doc.get("source", "").lower()
    payload = doc.get("payload", {})
    chunk_plan = str(payload.get("plan_name", "")).lower()

    # 1. Try to extract specific plan from question
    target_plan = _extract_plan_from_question(question)

    if target_plan:
        # Strict check: chunk must be from the right plan's document
        plan_words = target_plan.split()
        source_match = all(w in source for w in plan_words)
        payload_match = all(w in chunk_plan for w in plan_words)
        return source_match or payload_match

    # 2. Category-level question (e.g., "What types of endowment plans...")
    q_lower = question.lower()
    for cat_phrase, file_kw in _CATEGORY_KEYWORDS.items():
        if cat_phrase in q_lower:
            return file_kw in source or file_kw in chunk_plan

    # 3. Fallback: use Plan column + keyword overlap (for unknown question types)
    plan_kws = [w for w in plan.lower().split() if len(w) > 3]
    if plan_kws:
        plan_hit = any(kw in source or kw in chunk_plan for kw in plan_kws)
        if not plan_hit:
            return False

    # Require meaningful keyword overlap with expected answer
    kw_ov = keyword_overlap(expected, text)
    return kw_ov > 0.15


async def test_retrieval():
    """
    Test retrieval quality: checks if retrieved chunks are relevant at k=1,3,5,10.
    Reports Hit@k, MRR, and avg keyword overlap per config.
    """
    from app.services.rag.retriever import Retriever
    from app.services.rag.query_transformer import QueryTransformer

    print_header("TEST 2: RETRIEVAL QUALITY (CHUNK RELEVANCE)")

    questions = load_qa_dataset()
    if not questions:
        print("  No Q&A dataset found. Skipping.")
        return {}

    print(f"  Loaded {len(questions)} questions")

    retriever = Retriever()
    transformer = QueryTransformer()

    K_VALUES = [1, 3, 5, 10]
    MAX_K = max(K_VALUES)

    configs = [
        {"name": "Vector Only", "use_hybrid": False, "use_hyde": False, "use_decompose": False},
        {"name": "Hybrid (Vec+BM25)", "use_hybrid": True, "use_hyde": False, "use_decompose": False},
        {"name": "Vector + HyDE", "use_hybrid": False, "use_hyde": True, "use_decompose": False},
        {"name": "Hybrid + HyDE", "use_hybrid": True, "use_hyde": True, "use_decompose": False},
        {"name": "Vector + Decompose", "use_hybrid": False, "use_hyde": False, "use_decompose": True},
        {"name": "Hybrid + Decompose", "use_hybrid": True, "use_hyde": False, "use_decompose": True},
    ]

    all_metrics = {}
    all_detailed = {}  # config_name -> list of per-query details

    for cfg in configs:
        print(f"\n  --- Config: {cfg['name']} ---")
        # Per-query results
        hits_at_k = {k: 0 for k in K_VALUES}  # count of queries with >= 1 relevant chunk in top-k
        reciprocal_ranks = []
        kw_overlaps_at_k = {k: [] for k in K_VALUES}
        latencies = []
        detailed_results = []

        for i, q in enumerate(questions, 1):
            query = q["question"]
            expected = q.get("expected", "")
            plan = q.get("plan", "Unknown")

            print(f"    [{i}/{len(questions)}] {query[:55]}...", end=" ", flush=True)
            start = time.time()

            try:
                # Determine queries to run
                if cfg.get("use_decompose"):
                    sub_queries = transformer.decompose_query(query)
                else:
                    sub_queries = [query]

                all_docs = []
                for sq in sub_queries:
                    search_text = sq
                    is_hyde = False

                    if cfg["use_hyde"]:
                        hypo_doc = transformer.generate_hyde_doc(sq)
                        if hypo_doc:
                            search_text = hypo_doc
                            is_hyde = True

                    results = await retriever.search(
                        search_text,
                        limit=MAX_K,
                        use_hybrid=cfg["use_hybrid"],
                        is_hyde=is_hyde,
                        use_reranker=False,
                    )
                    all_docs.extend(results)

                # Deduplicate by id (keep first occurrence = highest ranked)
                seen_ids = set()
                docs = []
                for d in all_docs:
                    did = d.get("id") or d.get("text", "")[:100]
                    if did not in seen_ids:
                        seen_ids.add(did)
                        docs.append(d)

                elapsed = time.time() - start
                latencies.append(elapsed)

                # Find first relevant rank (for MRR)
                first_relevant_rank = None
                for rank, doc in enumerate(docs, 1):
                    if _check_chunk_relevant(doc, expected, plan, question=query):
                        if first_relevant_rank is None:
                            first_relevant_rank = rank
                        break

                # MRR
                rr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
                reciprocal_ranks.append(rr)

                # Hit@k and keyword overlap at each k
                for k in K_VALUES:
                    top_k = docs[:k]
                    has_hit = any(_check_chunk_relevant(d, expected, plan, question=query) for d in top_k)
                    if has_hit:
                        hits_at_k[k] += 1
                    combined_text = " ".join(d.get("text", "").lower() for d in top_k)
                    kw_overlaps_at_k[k].append(keyword_overlap(expected, combined_text))

                top_score = docs[0].get("score", 0) if docs else 0
                status = "HIT" if first_relevant_rank else "MISS"
                rank_str = f"@{first_relevant_rank}" if first_relevant_rank else ""
                print(f"[{status}{rank_str}] {elapsed:.1f}s score={top_score:.3f}")

                # Save detailed per-query result
                detailed_results.append({
                    "question": query,
                    "plan": plan,
                    "expected_answer": expected,
                    "status": status,
                    "first_relevant_rank": first_relevant_rank,
                    "latency": round(elapsed, 2),
                    "sub_queries": sub_queries if cfg.get("use_decompose") else None,
                    "retrieved_chunks": [
                        {
                            "rank": r + 1,
                            "source": d.get("source", ""),
                            "score": round(d.get("score", 0), 4),
                            "section_type": d.get("payload", {}).get("section_type", ""),
                            "chunk_title": d.get("payload", {}).get("chunk_title", ""),
                            "relevant": _check_chunk_relevant(d, expected, plan, question=query),
                            "text": d.get("text", "")[:500],
                        }
                        for r, d in enumerate(docs[:MAX_K])
                    ],
                })

            except Exception as e:
                elapsed = time.time() - start
                latencies.append(elapsed)
                reciprocal_ranks.append(0.0)
                for k in K_VALUES:
                    kw_overlaps_at_k[k].append(0.0)
                print(f"[ERR] {str(e)[:40]}")
                detailed_results.append({
                    "question": query, "plan": plan,
                    "expected_answer": expected,
                    "status": "ERR", "error": str(e),
                    "retrieved_chunks": [],
                })

        all_detailed[cfg["name"]] = detailed_results

        # Compute metrics
        total = len(questions)
        metrics = {
            "config": cfg["name"],
            "total": total,
            "MRR": round(sum(reciprocal_ranks) / max(total, 1), 3),
            "avg_latency": round(sum(latencies) / max(len(latencies), 1), 3),
        }
        for k in K_VALUES:
            metrics[f"Hit@{k}"] = round(100 * hits_at_k[k] / max(total, 1), 1)
            metrics[f"KW_Overlap@{k}"] = round(
                sum(kw_overlaps_at_k[k]) / max(len(kw_overlaps_at_k[k]), 1), 3
            )

        all_metrics[cfg["name"]] = metrics

        print(f"\n    MRR: {metrics['MRR']} | ", end="")
        print(" | ".join(f"Hit@{k}: {metrics[f'Hit@{k}']}%" for k in K_VALUES))

    # Comparison table
    print(f"\n  {'='*85}")
    print(f"  RETRIEVAL COMPARISON")
    print(f"  {'='*85}")
    header = f"  {'Config':<22} {'MRR':>5}"
    for k in K_VALUES:
        header += f" {'Hit@'+str(k):>7}"
    header += f" {'KWOv@5':>7} {'Latency':>8}"
    print(header)
    print(f"  {'-'*85}")
    for name, m in all_metrics.items():
        row = f"  {name:<22} {m['MRR']:>5.3f}"
        for k in K_VALUES:
            row += f" {m[f'Hit@{k}']:>6.1f}%"
        row += f" {m.get('KW_Overlap@5', 0):>7.3f}"
        row += f" {m['avg_latency']:>7.3f}s"
        print(row)

    # Save detailed results for the best config (by Hit@5)
    best_config = max(all_metrics, key=lambda n: all_metrics[n].get("Hit@5", 0))
    best_m = all_metrics[best_config]
    print(f"\n  Best config: {best_config} (Hit@5={best_m['Hit@5']}%)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = os.path.join(os.path.dirname(__file__), f"retrieval_details_{ts}.json")

    detail_output = {
        "timestamp": ts,
        "best_config": best_config,
        "metrics_summary": all_metrics,
        "questions": all_detailed[best_config],
    }

    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail_output, f, indent=2, ensure_ascii=False)

    print(f"  Detailed results saved: {detail_path}")

    return all_metrics


# ─────────────────────────────────────────────────────────────────
# Test Case 3: Reranking Quality
# ─────────────────────────────────────────────────────────────────

async def test_reranking():
    """Test if cross-encoder reranking improves result ordering."""
    from qdrant_client import QdrantClient
    from app.core.database import get_qdrant_client
    from app.core.model_cache import get_cached_embedder, get_cached_reranker

    print_header("TEST 3: RERANKING (CROSS-ENCODER)")

    questions = load_qa_dataset()
    if not questions:
        print("  No Q&A dataset found. Skipping.")
        return {}

    # Use a subset for reranking test
    subset = questions[:10]
    print(f"  Testing with {len(subset)} questions")

    qdrant = get_qdrant_client()
    embedder = get_cached_embedder()
    reranker = get_cached_reranker()

    print_metric("Reranker model:", reranker.model_name)
    print_metric("Expected model:", settings.RERANKER_MODEL)
    model_match = reranker.model_name == settings.RERANKER_MODEL
    print_metric("Model matches config:", "YES" if model_match else "NO - MISMATCH")

    changed_top1 = 0
    improved = 0
    rerank_times = []

    for i, q in enumerate(subset, 1):
        query = q["question"]
        expected = q.get("expected", "")
        print(f"  [{i}/{len(subset)}] {query[:55]}...", end=" ", flush=True)

        try:
            # Get raw vector results (no reranking)
            query_vec = embedder.embed_query(query)
            raw_results = qdrant.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query=query_vec,
                limit=10,
                with_payload=True,
            ).points

            if not raw_results:
                print("[NO RESULTS]")
                continue

            raw_docs = [
                {"id": str(h.id), "text": h.payload.get("text", ""), "source": h.payload.get("source", ""), "score": float(h.score)}
                for h in raw_results
            ]

            # Rerank
            rerank_start = time.time()
            reranked_docs = reranker.rerank(query, raw_docs, top_n=5)
            rerank_time = time.time() - rerank_start
            rerank_times.append(rerank_time)

            # Compare top-1
            raw_top1_text = raw_docs[0]["text"][:200] if raw_docs else ""
            reranked_top1_text = reranked_docs[0]["text"][:200] if reranked_docs else ""

            top1_changed = raw_top1_text != reranked_top1_text
            if top1_changed:
                changed_top1 += 1

            # Check if reranking improved relevance
            raw_overlap = keyword_overlap(expected, raw_top1_text)
            reranked_overlap = keyword_overlap(expected, reranked_top1_text)
            if reranked_overlap > raw_overlap:
                improved += 1

            status = "CHANGED" if top1_changed else "SAME"
            print(f"[{status}] rerank={rerank_time:.2f}s raw_overlap={raw_overlap:.2f} reranked_overlap={reranked_overlap:.2f}")

        except Exception as e:
            print(f"[ERR] {str(e)[:40]}")

    total = len(subset)
    avg_rerank = sum(rerank_times) / max(len(rerank_times), 1)

    print(f"\n  --- Reranking Summary ---")
    print_metric("Top-1 changed:", f"{changed_top1}/{total} ({100*changed_top1//max(total,1)}%)")
    print_metric("Relevance improved:", f"{improved}/{total} ({100*improved//max(total,1)}%)")
    print_metric("Avg rerank latency:", f"{avg_rerank:.3f}s")
    print_metric("Model verified:", "PASS" if model_match else "FAIL")

    return {
        "top1_changed_pct": round(100 * changed_top1 / max(total, 1), 1),
        "improved_pct": round(100 * improved / max(total, 1), 1),
        "avg_rerank_latency": round(avg_rerank, 3),
        "model_match": model_match,
    }


# ─────────────────────────────────────────────────────────────────
# Test Case 4: Generation Quality
# ─────────────────────────────────────────────────────────────────

def _check_hallucination(answer: str, context_texts: List[str]) -> dict:
    """Check if the answer stays grounded in context or hallucinates.

    Returns dict with:
      - grounded_ratio: fraction of answer sentences that overlap with context
      - has_refusal: True if model correctly refused (no info in context)
      - citation_count: number of [Source: ...] citations found
    """
    import re

    # Check refusal
    refusal_phrases = [
        "i don't have sufficient information",
        "not in the provided documents",
        "no information in the provided",
        "cannot answer",
        "not available in the context",
    ]
    answer_lower = answer.lower()
    has_refusal = any(p in answer_lower for p in refusal_phrases)

    # Count citations
    citation_count = len(re.findall(r'\[Source:\s*[^\]]+\]', answer))

    # Sentence-level grounding check
    context_blob = " ".join(context_texts).lower()
    context_words = set(context_blob.split())

    # Split answer into sentences
    sentences = re.split(r'[.!?\n]', answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return {"grounded_ratio": 1.0, "has_refusal": has_refusal, "citation_count": citation_count}

    grounded = 0
    for sent in sentences:
        sent_words = set(sent.lower().split())
        # Remove common stop words
        stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to",
                "in", "for", "on", "with", "at", "by", "from", "and", "or", "not", "this",
                "that", "it", "its", "as", "can", "will", "may", "if", "has", "have", "had",
                "do", "does", "did", "but", "no", "so", "up", "out", "all", "also", "than",
                "then", "into", "over", "such", "only", "very", "just", "about", "which",
                "their", "there", "these", "those", "they", "them", "we", "you", "your",
                "under", "plan", "policy", "shall", "should", "would", "could", "per"}
        content_words = sent_words - stop
        if not content_words:
            grounded += 1
            continue
        overlap = content_words & context_words
        if len(overlap) / len(content_words) >= 0.5:
            grounded += 1

    return {
        "grounded_ratio": round(grounded / len(sentences), 3),
        "has_refusal": has_refusal,
        "citation_count": citation_count,
    }


async def test_generation():
    """Test LLM generation quality across top 2 retrieval configs."""
    from app.services.rag.service import get_rag_service

    print_header("TEST 4: GENERATION QUALITY")

    questions = load_qa_dataset()
    if not questions:
        print("  No Q&A dataset found. Skipping.")
        return {}

    service = get_rag_service()
    print_metric("LLM model:", settings.LLM_MODEL_NAME)
    print(f"  Testing all {len(questions)} questions\n")

    # Top 2 retrieval configs from eval
    gen_configs = [
        {"name": "Hybrid (Vec+BM25)", "use_hybrid": True, "use_hyde": False, "use_decompose": False},
        {"name": "Vector Only", "use_hybrid": False, "use_hyde": False, "use_decompose": False},
    ]

    all_config_results = {}
    all_detailed = {}

    for cfg in gen_configs:
        cfg_name = cfg["name"]
        print(f"\n  {'='*60}")
        print(f"  CONFIG: {cfg_name}")
        print(f"  {'='*60}")

        results = []

        for i, q in enumerate(questions, 1):
            query = q["question"]
            expected = q.get("expected", "")
            plan = q.get("plan", "Unknown")

            print(f"  [{i}/{len(questions)}] {query[:65]}...", end=" ", flush=True)
            start = time.time()

            try:
                result = await service.answer_query(
                    query=query,
                    use_hyde=cfg["use_hyde"],
                    use_decomposition=cfg["use_decompose"],
                    use_hybrid_search=cfg["use_hybrid"],
                    use_auto_filters=True,
                    limit=5,
                )
                elapsed = time.time() - start

                answer = result.get("answer", "")
                citations = result.get("citations", [])
                context_texts = [c.get("text", "") for c in citations]

                # Metrics
                kw_ov = keyword_overlap(expected, answer)
                halluc = _check_hallucination(answer, context_texts)

                # Source accuracy: does any citation come from the right document?
                target_plan = _extract_plan_from_question(query)
                citation_sources = [c.get("source", "").lower() for c in citations]
                if target_plan:
                    plan_words = target_plan.split()
                    source_correct = any(
                        all(w in src for w in plan_words) for src in citation_sources
                    )
                else:
                    plan_kws = [w for w in str(plan).lower().split() if len(w) > 3]
                    source_correct = any(
                        any(kw in src for kw in plan_kws) for src in citation_sources
                    ) if plan_kws else True  # no plan to check against

                results.append({
                    "question": query,
                    "plan": str(plan),
                    "expected_answer": expected,
                    "generated_answer": answer,
                    "kw_overlap": round(kw_ov, 3),
                    "grounded_ratio": halluc["grounded_ratio"],
                    "citation_count": halluc["citation_count"],
                    "has_refusal": halluc["has_refusal"],
                    "source_correct": source_correct,
                    "latency": round(elapsed, 2),
                    "retrieved_chunks": [
                        {
                            "rank": idx + 1,
                            "source": c.get("source", ""),
                            "score": round(c.get("score", 0), 4),
                            "text": c.get("text", "")[:500],
                        }
                        for idx, c in enumerate(citations[:5])
                    ],
                })

                status = "OK" if kw_ov > 0.1 else "LOW"
                print(f"[{status}] kw={kw_ov:.2f} grnd={halluc['grounded_ratio']:.2f} "
                      f"cite={halluc['citation_count']} src={'Y' if source_correct else 'N'} "
                      f"{elapsed:.1f}s")

            except Exception as e:
                elapsed = time.time() - start
                results.append({
                    "question": query, "plan": str(plan),
                    "expected_answer": expected,
                    "generated_answer": "",
                    "kw_overlap": 0, "grounded_ratio": 0,
                    "citation_count": 0, "has_refusal": False,
                    "source_correct": False, "latency": round(elapsed, 2),
                    "error": str(e),
                })
                print(f"[ERR] {str(e)[:50]}")

        # Config summary
        total = len(results)
        avg_kw = sum(r["kw_overlap"] for r in results) / max(total, 1)
        avg_grnd = sum(r["grounded_ratio"] for r in results) / max(total, 1)
        avg_cite = sum(r["citation_count"] for r in results) / max(total, 1)
        src_rate = sum(1 for r in results if r["source_correct"]) / max(total, 1)
        avg_lat = sum(r["latency"] for r in results) / max(total, 1)
        refusal_ct = sum(1 for r in results if r["has_refusal"])

        print(f"\n  --- {cfg_name} Summary ---")
        print_metric("Avg KW overlap:", f"{avg_kw:.3f}")
        print_metric("Avg grounded ratio:", f"{avg_grnd:.3f}")
        print_metric("Avg citations/answer:", f"{avg_cite:.1f}")
        print_metric("Source accuracy:", f"{100*src_rate:.1f}%")
        print_metric("Refusals:", f"{refusal_ct}/{total}")
        print_metric("Avg latency:", f"{avg_lat:.1f}s")

        all_config_results[cfg_name] = {
            "avg_kw_overlap": round(avg_kw, 3),
            "avg_grounded_ratio": round(avg_grnd, 3),
            "avg_citations": round(avg_cite, 1),
            "source_accuracy": round(src_rate, 3),
            "refusals": refusal_ct,
            "avg_latency": round(avg_lat, 1),
        }
        all_detailed[cfg_name] = results

    # Comparison table
    print(f"\n  {'='*75}")
    print(f"  GENERATION COMPARISON")
    print(f"  {'='*75}")
    print(f"  {'Config':<25} {'KW_Ov':>6} {'Ground':>7} {'Cites':>6} {'SrcAcc':>7} {'Refuse':>7} {'Latency':>8}")
    print(f"  {'-'*75}")
    for cfg_name, m in all_config_results.items():
        print(f"  {cfg_name:<25} {m['avg_kw_overlap']:>6.3f} {m['avg_grounded_ratio']:>7.3f} "
              f"{m['avg_citations']:>6.1f} {m['source_accuracy']*100:>6.1f}% "
              f"{m['refusals']:>7} {m['avg_latency']:>7.1f}s")

    # Save detailed results for the best config (by kw_overlap)
    best_cfg = max(all_config_results, key=lambda k: all_config_results[k]["avg_kw_overlap"])
    print(f"\n  Best config: {best_cfg}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = os.path.join(os.path.dirname(__file__), f"generation_details_{timestamp}.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "llm_model": settings.LLM_MODEL_NAME,
            "best_config": best_cfg,
            "metrics_summary": all_config_results,
            "questions": all_detailed[best_cfg],
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Detailed results saved: {detail_path}")

    return all_config_results


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluation")
    parser.add_argument(
        "--test",
        choices=["all", "ingest", "chunking", "retrieval", "reranking", "generation"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of PDFs to ingest (0 = all)",
    )
    parser.add_argument(
        "--llm-metadata",
        action="store_true",
        default=False,
        help="Enable LLM-based metadata extraction during ingestion",
    )
    args = parser.parse_args()

    print_header(f"RAG PIPELINE EVALUATION - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print_metric("Embedding model:", settings.EMBEDDING_MODEL)
    print_metric("Reranker model:", settings.RERANKER_MODEL)
    print_metric("LLM provider:", f"{settings.LLM_PROVIDER} / {settings.LLM_MODEL_NAME}")
    print_metric("Chunk size:", settings.CHUNK_SIZE)

    all_results = {}

    if args.test in ("all", "ingest"):
        all_results["ingestion"] = await test_ingestion(limit=args.limit, use_llm_metadata=args.llm_metadata)

    if args.test in ("all", "chunking"):
        all_results["chunking"] = await test_chunking_and_metadata()

    if args.test in ("all", "retrieval"):
        all_results["retrieval"] = await test_retrieval()

    if args.test in ("all", "reranking"):
        all_results["reranking"] = await test_reranking()

    if args.test in ("all", "generation"):
        all_results["generation"] = await test_generation()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(os.path.dirname(__file__), f"eval_results_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print_header("EVALUATION COMPLETE")
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
