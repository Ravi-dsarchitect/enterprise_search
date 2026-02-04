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

async def test_ingestion(limit: int = 0):
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
    service = IngestionService(use_layout_aware=True, use_llm_metadata=False)
    total_chunks = 0
    results = []

    for i, pdf_path in enumerate(pdf_files, 1):
        fname = os.path.basename(pdf_path)
        print(f"  [{i}/{len(pdf_files)}] {fname[:50]}...", end=" ", flush=True)
        start = time.time()
        try:
            result = await service.process_local_file(pdf_path)
            elapsed = time.time() - start
            chunks = result.get("chunks", 0)
            total_chunks += chunks
            results.append({"file": fname, "chunks": chunks, "time": elapsed, "status": "OK"})
            print(f"[OK] {chunks} chunks ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - start
            results.append({"file": fname, "chunks": 0, "time": elapsed, "status": f"FAIL: {e}"})
            print(f"[FAIL] {str(e)[:50]}")

    # Rebuild BM25 index after ingestion
    from app.core.database import rebuild_bm25_index
    print("\n  Rebuilding BM25 index...")
    rebuild_bm25_index()

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
    """Test chunk quality and 3-layer metadata integration."""
    from app.services.ingestion.parsers import DocumentParserFactory
    from app.services.ingestion.chunkers import HybridLayoutSemanticChunker
    from app.services.ingestion.metadata import (
        enrich_chunk_metadata,
        extract_structured_fields,
        SpacyExtractor,
        MetadataExtractor,
        get_section_display_name,
    )
    from app.core.model_cache import get_cached_embedder

    print_header("TEST 1: CHUNKING + 3-LAYER METADATA")

    if not os.path.exists(SAMPLE_PDF):
        print(f"  Sample PDF not found: {SAMPLE_PDF}")
        return {}

    # 1. Parse
    print(f"  Parsing: {os.path.basename(SAMPLE_PDF)}")
    parser = DocumentParserFactory.get_parser(SAMPLE_PDF)
    text = parser.parse(SAMPLE_PDF)
    print_metric("Extracted text length:", f"{len(text)} chars")

    # 2. Chunk
    print("\n  Running HybridLayoutSemanticChunker...")
    embedder = get_cached_embedder()
    chunker = HybridLayoutSemanticChunker(embedder=embedder, chunk_size=1800)
    chunks = chunker.chunk_with_metadata(text)

    sizes = [len(c.text) for c in chunks]
    print_metric("Chunks generated:", len(chunks))
    print_metric("Avg chunk size:", f"{sum(sizes) // len(sizes)} chars")
    print_metric("Min / Max chunk size:", f"{min(sizes)} / {max(sizes)} chars")

    # Section distribution
    section_counts = defaultdict(int)
    for c in chunks:
        section_counts[c.section_type] += 1
    non_general = sum(1 for c in chunks if c.section_type != "general")
    print_metric("Chunks with section != general:", f"{non_general}/{len(chunks)} ({100*non_general//len(chunks)}%)")

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
        for field in ["plan_number", "uin", "age_ranges", "monetary_amounts", "benefit_types"]:
            if meta.get(field):
                layer1_fields_found.add(field)

    print_metric("Chunks with tags:", f"{layer1_tag_count}/{len(chunks)} ({100*layer1_tag_count//len(chunks)}%)")
    print_metric("Fields found:", ", ".join(layer1_fields_found) if layer1_fields_found else "(none)")

    # 4. Layer 2: spaCy NER
    print("\n  --- Layer 2: spaCy NER ---")
    ner_total = defaultdict(int)
    chunks_with_ner = 0
    for c in chunks[:20]:  # Sample first 20 chunks
        entities = SpacyExtractor.extract_entities(c.text)
        if entities:
            chunks_with_ner += 1
        for key, vals in entities.items():
            ner_total[key] += len(vals)

    sample_n = min(20, len(chunks))
    print_metric("Chunks with NER entities:", f"{chunks_with_ner}/{sample_n} (sampled)")
    for key, count in sorted(ner_total.items()):
        print(f"    {key:<25} {count} entities")

    # 5. Layer 3: LLM-based extraction
    print("\n  --- Layer 3: LLM-based extraction ---")
    layer3_success = False
    layer3_result = {}
    try:
        extractor = MetadataExtractor()
        layer3_result = extractor.extract_metadata(text, os.path.basename(SAMPLE_PDF))
        layer3_success = bool(layer3_result.get("plan_name") or layer3_result.get("plan_type"))
        print_metric("LLM extraction:", "SUCCESS" if layer3_success else "PARTIAL")
        for k, v in layer3_result.items():
            if v and k != "keywords":
                print(f"    {k:<25} {str(v)[:60]}")
        if layer3_result.get("keywords"):
            print(f"    {'keywords':<25} {layer3_result['keywords'][:5]}")
    except Exception as e:
        print_metric("LLM extraction:", f"FAILED ({e})")

    # 6. Sample enriched chunks
    print("\n  --- Sample Enriched Chunks (first 3) ---")
    for i, c in enumerate(chunks[:3]):
        meta = enrich_chunk_metadata(c.text, c.section_type, c.content_type)
        print(f"\n  Chunk {i+1}: [{c.section_type}] [{c.content_type}] (page {c.page_number})")
        print(f"    Tags:   {meta.get('chunk_tags', [])[:5]}")
        print(f"    Hints:  {meta.get('entity_hints', [])}")
        print(f"    Text:   {c.text[:150].replace(chr(10), ' ')}...")

    metrics = {
        "chunk_count": len(chunks),
        "avg_size": sum(sizes) // len(sizes),
        "pct_with_section": round(100 * non_general / len(chunks), 1),
        "pct_with_tags": round(100 * layer1_tag_count / len(chunks), 1),
        "layer1_fields": list(layer1_fields_found),
        "layer2_ner_count": dict(ner_total),
        "layer3_success": layer3_success,
    }

    print(f"\n  TEST 1 RESULT: {'PASS' if layer1_tag_count > 0 and chunks_with_ner > 0 else 'PARTIAL'}")
    return metrics


# ─────────────────────────────────────────────────────────────────
# Test Case 2: Retrieval Quality
# ─────────────────────────────────────────────────────────────────

async def test_retrieval():
    """Test retrieval quality across 4 configurations."""
    from app.services.rag.retriever import Retriever
    from app.services.rag.query_transformer import QueryTransformer

    print_header("TEST 2: RETRIEVAL QUALITY")

    questions = load_qa_dataset()
    if not questions:
        print("  No Q&A dataset found. Skipping.")
        return {}

    print(f"  Loaded {len(questions)} questions")

    retriever = Retriever()
    transformer = QueryTransformer()

    configs = [
        {"name": "Vector Only", "use_hybrid": False, "use_hyde": False},
        {"name": "Hybrid (Vector+BM25)", "use_hybrid": True, "use_hyde": False},
        {"name": "Vector + HyDE", "use_hybrid": False, "use_hyde": True},
        {"name": "Hybrid + HyDE", "use_hybrid": True, "use_hyde": True},
    ]

    all_metrics = {}

    for cfg in configs:
        print(f"\n  --- Config: {cfg['name']} ---")
        results = []

        for i, q in enumerate(questions, 1):
            query = q["question"]
            expected = q.get("expected", "")
            plan = q.get("plan", "Unknown")

            print(f"    [{i}/{len(questions)}] {query[:55]}...", end=" ", flush=True)
            start = time.time()

            try:
                search_text = query
                is_hyde = False

                if cfg["use_hyde"]:
                    hypo_doc = transformer.generate_hyde_doc(query)
                    if hypo_doc:
                        search_text = hypo_doc
                        is_hyde = True

                docs = await retriever.search(
                    search_text,
                    limit=5,
                    use_hybrid=cfg["use_hybrid"],
                    is_hyde=is_hyde,
                    use_reranker=False,  # Pure retrieval — reranking tested separately in Test 3
                )
                elapsed = time.time() - start

                # Evaluate
                top_score = docs[0].get("score", 0) if docs else 0
                sources = " ".join(d.get("source", "").lower() for d in docs[:3])
                texts = " ".join(d.get("text", "").lower() for d in docs[:3])
                plan_kws = [w for w in plan.lower().split() if len(w) > 3]
                plan_match = any(kw in sources or kw in texts for kw in plan_kws)
                kw_overlap = keyword_overlap(expected, texts)

                is_relevant = top_score >= 0.5 or (plan_match and kw_overlap > 0.15)

                results.append({
                    "query": query, "plan": plan,
                    "top_score": round(top_score, 4),
                    "is_relevant": is_relevant, "latency": elapsed,
                    "plan_match": plan_match, "kw_overlap": round(kw_overlap, 3),
                })

                status = "OK" if is_relevant else "FAIL"
                print(f"[{status}] {elapsed:.1f}s score={top_score:.3f}")

            except Exception as e:
                elapsed = time.time() - start
                results.append({
                    "query": query, "plan": plan,
                    "top_score": 0, "is_relevant": False,
                    "latency": elapsed, "error": str(e),
                })
                print(f"[ERR] {str(e)[:40]}")

        # Metrics
        total = len(results)
        successful = sum(1 for r in results if r["is_relevant"])
        scores = [r["top_score"] for r in results]
        latencies = [r["latency"] for r in results]

        metrics = {
            "config": cfg["name"],
            "total": total,
            "success_rate": round(100 * successful / max(total, 1), 1),
            "avg_top_score": round(sum(scores) / max(len(scores), 1), 3),
            "avg_latency": round(sum(latencies) / max(len(latencies), 1), 3),
        }
        all_metrics[cfg["name"]] = metrics

        print(f"\n    Success: {metrics['success_rate']}% | Avg Score: {metrics['avg_top_score']} | Avg Latency: {metrics['avg_latency']}s")

    # Comparison table
    print("\n  --- Retrieval Comparison ---")
    print(f"  {'Config':<25} {'Success%':>10} {'AvgScore':>10} {'Latency':>10}")
    print(f"  {'-'*55}")
    for name, m in all_metrics.items():
        print(f"  {name:<25} {m['success_rate']:>9.1f}% {m['avg_top_score']:>10.3f} {m['avg_latency']:>9.3f}s")

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

async def test_generation():
    """Test LLM generation quality with citations."""
    from app.services.rag.service import get_rag_service

    print_header("TEST 4: GENERATION QUALITY")

    questions = load_qa_dataset()
    if not questions:
        print("  No Q&A dataset found. Skipping.")
        return {}

    # Use subset for generation (LLM calls are slow)
    subset = questions[:5]
    print(f"  Testing with {len(subset)} questions (LLM generation)")

    service = get_rag_service()
    results = []

    for i, q in enumerate(subset, 1):
        query = q["question"]
        expected = q.get("expected", "")
        plan = q.get("plan", "Unknown")

        print(f"\n  [{i}/{len(subset)}] {query[:60]}...")
        start = time.time()

        try:
            result = await service.answer_query(
                query=query,
                use_hyde=False,
                use_decomposition=False,
                use_hybrid_search=True,
                use_auto_filters=False,
                limit=5,
            )
            elapsed = time.time() - start

            answer = result.get("answer", "")
            citations = result.get("citations", [])

            # Metrics
            kw_overlap = keyword_overlap(expected, answer)
            has_citations = len(citations) > 0
            plan_kws = [w for w in plan.lower().split() if len(w) > 3]
            citation_sources = [c.get("source", "").lower() for c in citations]
            source_match = any(
                any(kw in src for kw in plan_kws)
                for src in citation_sources
            )

            results.append({
                "query": query,
                "plan": plan,
                "kw_overlap": round(kw_overlap, 3),
                "has_citations": has_citations,
                "source_match": source_match,
                "latency": elapsed,
                "answer_preview": answer[:200],
            })

            print(f"    Accuracy (kw overlap): {kw_overlap:.2f}")
            print(f"    Citations: {len(citations)} | Source match: {source_match}")
            print(f"    Latency: {elapsed:.1f}s")
            print(f"    Answer: {answer[:150].replace(chr(10), ' ')}...")

        except Exception as e:
            elapsed = time.time() - start
            results.append({
                "query": query, "plan": plan,
                "kw_overlap": 0, "has_citations": False,
                "source_match": False, "latency": elapsed,
                "error": str(e),
            })
            print(f"    [ERROR] {str(e)[:60]}")

    # Test streaming (1 query)
    print("\n  --- Streaming Test ---")
    stream_pass = False
    try:
        stream_q = subset[0]["question"]
        token_count = 0
        async for event in service.answer_query_stream(
            query=stream_q, use_hybrid_search=False, use_auto_filters=False, limit=3
        ):
            if event.get("type") == "token":
                token_count += 1
            if event.get("type") == "done":
                stream_pass = True
        print(f"    Streaming: PASS ({token_count} tokens received)")
    except Exception as e:
        print(f"    Streaming: FAIL ({e})")

    # Summary
    total = len(results)
    avg_kw = sum(r["kw_overlap"] for r in results) / max(total, 1)
    citation_rate = sum(1 for r in results if r["has_citations"]) / max(total, 1)
    source_rate = sum(1 for r in results if r["source_match"]) / max(total, 1)
    avg_latency = sum(r["latency"] for r in results) / max(total, 1)

    print(f"\n  --- Generation Summary ---")
    print_metric("Avg keyword overlap:", f"{avg_kw:.3f}")
    print_metric("Citation presence:", f"{100*citation_rate:.0f}%")
    print_metric("Source accuracy:", f"{100*source_rate:.0f}%")
    print_metric("Avg generation latency:", f"{avg_latency:.1f}s")
    print_metric("Streaming test:", "PASS" if stream_pass else "FAIL")

    return {
        "avg_kw_overlap": round(avg_kw, 3),
        "citation_rate": round(citation_rate, 3),
        "source_accuracy": round(source_rate, 3),
        "avg_latency": round(avg_latency, 1),
        "streaming_pass": stream_pass,
    }


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
    args = parser.parse_args()

    print_header(f"RAG PIPELINE EVALUATION - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print_metric("Embedding model:", settings.EMBEDDING_MODEL)
    print_metric("Reranker model:", settings.RERANKER_MODEL)
    print_metric("LLM provider:", f"{settings.LLM_PROVIDER} / {settings.LLM_MODEL_NAME}")
    print_metric("Chunk size:", settings.CHUNK_SIZE)

    all_results = {}

    if args.test in ("all", "ingest"):
        all_results["ingestion"] = await test_ingestion(limit=args.limit)

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
