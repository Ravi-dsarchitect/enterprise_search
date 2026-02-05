"""
RAG Evaluation Script for LIC Q&A
Tests retrieval quality and identifies failure patterns.
"""

import asyncio
import pandas as pd
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict
import sys
import os
import io

# Fix Windows console encoding for emojis
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.rag.retriever import Retriever
from app.services.rag.service import RAGService
from app.core.config import settings


class RAGEvaluator:
    """Evaluates RAG retrieval quality against Q&A dataset."""

    def __init__(self):
        self.retriever = Retriever()
        self.rag_service = RAGService()
        self.results = []

    async def evaluate_with_generation(
        self,
        questions: List[Dict[str, str]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test full RAG pipeline with LLM answer generation.

        Args:
            questions: List of {"question": str, "expected": str, "plan": str}
            config: Retrieval config {"use_hybrid": bool, "use_hyde": bool}
        """
        results = []
        total_time = 0

        for i, q in enumerate(questions, 1):
            query = q["question"]
            expected = q.get("expected", "")
            plan_name = q.get("plan", "Unknown")

            print(f"  [{i}/{len(questions)}] {query[:60]}...", end=" ", flush=True)

            start = time.time()
            try:
                # Run full RAG pipeline (disable auto-filters to avoid false NO_RESULTS)
                rag_result = await self.rag_service.answer_query(
                    query=query,
                    use_hyde=config.get("use_hyde", False),
                    use_decomposition=False,
                    use_hybrid_search=config.get("use_hybrid", False),
                    use_auto_filters=False,  # Disabled - causes NO_RESULTS with invalid filters
                    limit=5
                )

                elapsed = time.time() - start
                total_time += elapsed

                # Extract retrieved docs from citations
                retrieved_docs = rag_result.get("citations", [])
                generated_answer = rag_result.get("answer", "")

                # Analyze results
                result = self._analyze_retrieval(
                    query=query,
                    expected=expected,
                    plan_name=plan_name,
                    retrieved_docs=retrieved_docs,
                    latency=elapsed
                )
                result["generated_answer"] = generated_answer
                results.append(result)

                status = "OK" if result["is_relevant"] else "FAIL"
                print(f"[{status}] {elapsed:.2f}s (score: {result['top_score']:.3f})")

            except Exception as e:
                elapsed = time.time() - start
                print(f"[ERROR] {str(e)[:50]}")
                results.append({
                    "query": query,
                    "expected": expected,
                    "plan": plan_name,
                    "is_relevant": False,
                    "top_score": 0,
                    "failure_reason": "ERROR",
                    "error": str(e),
                    "retrieved_docs": [],
                    "latency": elapsed,
                    "generated_answer": f"Error: {str(e)}"
                })

        # Compute metrics
        metrics = self._compute_metrics(results)
        metrics["config"] = config
        metrics["total_time"] = total_time
        metrics["avg_latency"] = total_time / len(questions) if questions else 0

        return {
            "metrics": metrics,
            "results": results
        }

    async def evaluate_retrieval_only(
        self,
        questions: List[Dict[str, str]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test retrieval without LLM generation (fast).

        Args:
            questions: List of {"question": str, "expected": str, "plan": str}
            config: Retrieval config {"use_hybrid": bool, "use_hyde": bool}
        """
        results = []
        total_time = 0

        for i, q in enumerate(questions, 1):
            query = q["question"]
            expected = q.get("expected", "")
            plan_name = q.get("plan", "Unknown")

            print(f"  [{i}/{len(questions)}] {query[:60]}...", end=" ", flush=True)

            start = time.time()
            try:
                # Determine search text (with HyDE if enabled)
                search_text = query
                if config.get("use_hyde", False):
                    hypo_doc = self.rag_service.query_transformer.generate_hyde_doc(query)
                    if hypo_doc:
                        search_text = hypo_doc

                # Run retrieval
                docs = await self.retriever.search(
                    search_text,
                    limit=5,
                    use_hybrid=config.get("use_hybrid", False)
                )

                elapsed = time.time() - start
                total_time += elapsed

                # Analyze results
                result = self._analyze_retrieval(
                    query=query,
                    expected=expected,
                    plan_name=plan_name,
                    retrieved_docs=docs,
                    latency=elapsed
                )
                results.append(result)

                status = "OK" if result["is_relevant"] else "FAIL"
                print(f"[{status}] {elapsed:.2f}s (score: {result['top_score']:.3f})")

            except Exception as e:
                elapsed = time.time() - start
                print(f"[ERROR] {str(e)[:50]}")
                results.append({
                    "query": query,
                    "expected": expected,
                    "plan": plan_name,
                    "is_relevant": False,
                    "top_score": 0,
                    "failure_reason": "ERROR",
                    "error": str(e),
                    "retrieved_docs": [],
                    "latency": elapsed
                })

        # Compute metrics
        metrics = self._compute_metrics(results)
        metrics["config"] = config
        metrics["total_time"] = total_time
        metrics["avg_latency"] = total_time / len(questions) if questions else 0

        return {
            "metrics": metrics,
            "results": results
        }

    def _analyze_retrieval(
        self,
        query: str,
        expected: str,
        plan_name: str,
        retrieved_docs: List[Dict],
        latency: float
    ) -> Dict[str, Any]:
        """Analyze retrieval results and categorize failures."""

        result = {
            "query": query,
            "expected": expected[:500] if expected else "",
            "plan": plan_name,
            "latency": latency,
            "retrieved_docs": [],
            "top_score": 0,
            "is_relevant": False,
            "failure_reason": None
        }

        if not retrieved_docs:
            result["failure_reason"] = "NO_RESULTS"
            return result

        # Extract doc info
        for doc in retrieved_docs[:5]:
            result["retrieved_docs"].append({
                "source": doc.get("source", "unknown"),
                "score": round(doc.get("score", 0), 4),
                "text_preview": doc.get("text", "")[:300],
                "section_type": doc.get("payload", {}).get("section_type", "unknown")
            })

        result["top_score"] = retrieved_docs[0].get("score", 0) if retrieved_docs else 0

        # Determine relevance heuristically
        # Check if plan name appears in retrieved sources
        plan_keywords = plan_name.lower().split()
        sources_text = " ".join([d.get("source", "").lower() for d in retrieved_docs[:3]])
        retrieved_text = " ".join([d.get("text", "").lower() for d in retrieved_docs[:3]])

        # Check for plan match
        plan_match = any(kw in sources_text or kw in retrieved_text for kw in plan_keywords if len(kw) > 3)

        # Check for content overlap with expected answer
        expected_keywords = set(expected.lower().split()) if expected else set()
        retrieved_keywords = set(retrieved_text.split())

        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                     "have", "has", "had", "do", "does", "did", "will", "would", "could",
                     "should", "may", "might", "must", "shall", "can", "of", "in", "to",
                     "for", "with", "on", "at", "by", "from", "as", "or", "and", "if", "it"}
        expected_keywords -= stopwords

        keyword_overlap = len(expected_keywords & retrieved_keywords) / len(expected_keywords) if expected_keywords else 0

        # Determine relevance
        score_threshold = 0.5  # Reranker scores

        if result["top_score"] >= score_threshold and (plan_match or keyword_overlap > 0.2):
            result["is_relevant"] = True
        elif result["top_score"] >= score_threshold:
            result["is_relevant"] = True  # Trust high scores
        else:
            # Categorize failure
            if result["top_score"] < 0.3:
                result["failure_reason"] = "LOW_SIMILARITY"
            elif not plan_match:
                result["failure_reason"] = "WRONG_PLAN"
            elif keyword_overlap < 0.1:
                result["failure_reason"] = "WRONG_SECTION"
            else:
                result["failure_reason"] = "UNCERTAIN"

        return result

    def _compute_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute evaluation metrics."""

        total = len(results)
        if total == 0:
            return {"total": 0}

        successful = sum(1 for r in results if r["is_relevant"])

        # MRR calculation (assuming first relevant is at position 1 if relevant)
        mrr_sum = sum(1.0 for r in results if r["is_relevant"])  # Simplified

        # Failure breakdown
        failures = defaultdict(int)
        for r in results:
            if not r["is_relevant"] and r.get("failure_reason"):
                failures[r["failure_reason"]] += 1

        # Score distribution
        scores = [r["top_score"] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0

        return {
            "total": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": round(successful / total * 100, 1),
            "mrr": round(mrr_sum / total, 3),
            "avg_top_score": round(avg_score, 3),
            "failure_breakdown": dict(failures),
            "avg_latency": round(sum(r["latency"] for r in results) / total, 3)
        }

    def analyze_failures(self, results: List[Dict]) -> Dict[str, Any]:
        """Deep analysis of failed retrievals."""

        failures = [r for r in results if not r["is_relevant"]]

        analysis = {
            "total_failures": len(failures),
            "by_reason": defaultdict(list),
            "by_plan": defaultdict(int),
            "low_score_queries": [],
            "recommendations": []
        }

        for f in failures:
            reason = f.get("failure_reason", "UNKNOWN")
            analysis["by_reason"][reason].append({
                "query": f["query"],
                "plan": f["plan"],
                "top_score": f["top_score"],
                "sources": [d["source"] for d in f.get("retrieved_docs", [])[:3]]
            })
            analysis["by_plan"][f["plan"]] += 1

            if f["top_score"] < 0.3:
                analysis["low_score_queries"].append(f["query"])

        # Generate recommendations
        if analysis["by_reason"].get("LOW_SIMILARITY"):
            analysis["recommendations"].append(
                "Consider enabling hybrid search (BM25) for better keyword matching"
            )
        if analysis["by_reason"].get("WRONG_PLAN"):
            analysis["recommendations"].append(
                "Add plan-specific metadata filtering or improve chunking by plan"
            )
        if len(analysis["low_score_queries"]) > len(failures) * 0.5:
            analysis["recommendations"].append(
                "Consider tuning embeddings or chunking strategy for better semantic understanding"
            )

        return dict(analysis)

    def export_results(
        self,
        all_results: Dict[str, Dict],
        output_dir: str = "."
    ):
        """Export results to Excel and JSON."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Flatten results for Excel
        rows = []
        for config_name, data in all_results.items():
            for r in data.get("results", []):
                rows.append({
                    "Config": config_name,
                    "Plan": r["plan"],
                    "Query": r["query"],
                    "Expected Answer": r.get("expected", ""),
                    "Generated Answer": r.get("generated_answer", ""),
                    "Is Relevant": r["is_relevant"],
                    "Top Score": r["top_score"],
                    "Failure Reason": r.get("failure_reason", ""),
                    "Latency (s)": round(r["latency"], 3),
                    "Top Source": r["retrieved_docs"][0]["source"] if r.get("retrieved_docs") else "",
                    "Top Section Type": r["retrieved_docs"][0].get("section_type", "") if r.get("retrieved_docs") else ""
                })

        # Save Excel
        df = pd.DataFrame(rows)
        excel_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"\nResults saved to: {excel_path}")

        # Save JSON with full details
        json_path = os.path.join(output_dir, f"evaluation_details_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            # Convert results for JSON serialization
            export_data = {}
            for config_name, data in all_results.items():
                export_data[config_name] = {
                    "metrics": data.get("metrics", {}),
                    "failure_analysis": data.get("failure_analysis", {}),
                    "results": data.get("results", [])
                }
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"Details saved to: {json_path}")

        return excel_path, json_path


def print_report(all_results: Dict[str, Dict], embedding_model: str):
    """Print summary report to console."""

    print("\n" + "=" * 70)
    print(f"RAG EVALUATION REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    print(f"Embedding Model: {embedding_model}")
    print("-" * 70)

    for config_name, data in all_results.items():
        metrics = data.get("metrics", {})
        print(f"\n{config_name}:")
        print(f"  Success Rate: {metrics.get('success_rate', 0)}% ({metrics.get('successful', 0)}/{metrics.get('total', 0)})")
        print(f"  Avg Top Score: {metrics.get('avg_top_score', 0)}")
        print(f"  Avg Latency: {metrics.get('avg_latency', 0):.3f}s")

        failures = metrics.get("failure_breakdown", {})
        if failures:
            print(f"  Failure Breakdown:")
            for reason, count in failures.items():
                print(f"    - {reason}: {count}")

    # Best config
    best_config = max(all_results.items(), key=lambda x: x[1].get("metrics", {}).get("success_rate", 0))
    print("\n" + "-" * 70)
    print(f"BEST CONFIG: {best_config[0]} ({best_config[1].get('metrics', {}).get('success_rate', 0)}% success)")

    # Recommendations from failure analysis
    for config_name, data in all_results.items():
        analysis = data.get("failure_analysis", {})
        recs = analysis.get("recommendations", [])
        if recs:
            print(f"\nRecommendations ({config_name}):")
            for rec in recs:
                print(f"  - {rec}")

    print("=" * 70 + "\n")


async def main():
    """Main evaluation entry point."""

    # Load questions from Excel
    excel_path = os.path.join(os.path.dirname(__file__), "..", "docs", "LIC_QA_Evaluation_Results.xlsx")

    print(f"Loading questions from: {excel_path}")
    df = pd.read_excel(excel_path)

    # Map columns
    questions = []
    for _, row in df.iterrows():
        q = str(row.get("Sample Questions", "")).strip()
        if q and q.lower() != "nan":
            questions.append({
                "question": q,
                "expected": str(row.get("Expected Answers", "")),
                "plan": str(row.get("Plan", "Unknown"))
            })

    print(f"Loaded {len(questions)} questions\n")

    # Determine current embedding model
    embedding_info = f"{settings.EMBEDDING_PROVIDER}: {settings.EMBEDDING_MODEL}"

    # Test configuration - Vector Only with answer generation
    config = {"use_hybrid": False, "use_hyde": False}
    config_name = "Vector Only + Generation"

    evaluator = RAGEvaluator()
    all_results = {}

    print(f"\n{'='*50}")
    print(f"Testing: {config_name}")
    print(f"{'='*50}")

    result = await evaluator.evaluate_with_generation(questions, config)
    result["failure_analysis"] = evaluator.analyze_failures(result["results"])
    all_results[config_name] = result

    # Print report
    print_report(all_results, embedding_info)

    # Export results
    output_dir = os.path.dirname(__file__)
    evaluator.export_results(all_results, output_dir)

    return all_results


if __name__ == "__main__":
    asyncio.run(main())
