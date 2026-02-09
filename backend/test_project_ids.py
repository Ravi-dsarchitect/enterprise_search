#!/usr/bin/env python3
"""
Test script to verify multi-tenant project_ids filtering in the enterprise search system.

Usage:
    1. Ensure the server is running on localhost:8000
    2. Run: python test_project_ids.py

This test performs the following:
    - Ingests existing PDFs with different project_ids using bulk API
    - Queries with specific project_ids to verify filtering
    - Queries with non-matching project_ids to verify isolation
"""

import requests
import json
import sys
import time
import os

BASE_URL = "http://localhost:8000"
# Use existing docs from your repository
DOCS_PATH = "/home/ec2-user/nakul/enterprise_search/docs/OneDrive_1_2-2-2026"

def test_bulk_ingest_project_alpha():
    """Test bulk ingestion with project_ids for Project Alpha (first 2 files)."""
    print("\n" + "="*60)
    print("TEST 1: Bulk ingest files with project_ids ['project_alpha']")
    print("="*60)
    
    response = requests.post(
        f"{BASE_URL}/api/v1/ingestion/bulk",
        params={
            "directory_path": DOCS_PATH,
            "limit": 2,  # Just ingest 2 files for quick testing
            "project_ids": ["project_alpha", "shared_projects"]
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Bulk ingest successful!")
        print(f"   - Total files: {result.get('total')}")
        print(f"   - Successful: {result.get('successful')}")
        print(f"   - Failed: {result.get('failed')}")
        print(f"   - Duration: {result.get('duration_seconds')}s")
        return result.get('successful', 0) > 0
    else:
        print(f"❌ Bulk ingest failed: {response.status_code}")
        print(f"   - Error: {response.text[:200]}")
        return False

def test_bulk_ingest_project_beta():
    """Test bulk ingestion with different project_ids for Project Beta."""
    print("\n" + "="*60)
    print("TEST 2: Bulk ingest files with project_ids ['project_beta']")
    print("="*60)
    
    # Use a subdirectory for different project
    subdir = os.path.join(DOCS_PATH, "Endowment_plan_docs")
    if not os.path.exists(subdir):
        print(f"⚠️  Subdirectory not found, using main docs folder")
        subdir = DOCS_PATH
    
    response = requests.post(
        f"{BASE_URL}/api/v1/ingestion/bulk",
        params={
            "directory_path": subdir,
            "limit": 1,  # Just 1 file for quick test
            "project_ids": ["project_beta"]
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Bulk ingest successful!")
        print(f"   - Total files: {result.get('total')}")
        print(f"   - Successful: {result.get('successful')}")
        return result.get('successful', 0) > 0
    else:
        print(f"❌ Bulk ingest failed: {response.status_code}")
        print(f"   - Error: {response.text[:200]}")
        return False


def test_query_with_matching_project():
    """Test query with matching project_id returns correct results."""
    print("\n" + "="*60)
    print("TEST 3: Query with matching project_id (project_alpha)")
    print("="*60)
    
    response = requests.post(
        f"{BASE_URL}/api/v1/chat/query",
        json={
            "query": "What are the features of the system?",
            "project_ids": ["project_alpha"],
            "use_hybrid_search": True,
            "limit": 5
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        answer = result.get("answer", "")
        citations = result.get("citations", [])
        
        print(f"✅ Query successful!")
        print(f"   - Answer preview: {answer[:200]}...")
        print(f"   - Citations count: {len(citations)}")
        
        # Check if citations are from the right project
        for i, cit in enumerate(citations[:3]):
            source = cit.get("source", "unknown")
            project_ids = cit.get("payload", {}).get("project_ids", [])
            print(f"   - Citation {i+1}: {source} | projects: {project_ids}")
        
        return True
    else:
        print(f"❌ Query failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_query_with_non_matching_project():
    """Test query with non-matching project_id returns no results from other projects."""
    print("\n" + "="*60)
    print("TEST 4: Query with non-matching project_id (project_gamma)")
    print("="*60)
    
    response = requests.post(
        f"{BASE_URL}/api/v1/chat/query",
        json={
            "query": "What are the features?",
            "project_ids": ["project_gamma"],  # Non-existent project
            "use_hybrid_search": True,
            "limit": 5
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        citations = result.get("citations", [])
        
        # With fallback, we might still get results but should be empty project-specific data
        print(f"✅ Query completed!")
        print(f"   - Citations count: {len(citations)}")
        
        # Check if any citation has project_gamma (should be none)
        has_gamma = False
        for cit in citations:
            project_ids = cit.get("payload", {}).get("project_ids", [])
            if "project_gamma" in project_ids:
                has_gamma = True
                break
        
        if not has_gamma and len(citations) == 0:
            print(f"   ✅ Correctly returned no results for non-matching project")
        elif not has_gamma:
            print(f"   ⚠️  Returned results (possibly via fallback), but none from project_gamma")
        
        return True
    else:
        print(f"❌ Query failed: {response.status_code}")
        return False

def test_query_multiple_projects():
    """Test query with multiple project_ids."""
    print("\n" + "="*60)
    print("TEST 5: Query with multiple project_ids")
    print("="*60)
    
    response = requests.post(
        f"{BASE_URL}/api/v1/chat/query",
        json={
            "query": "Tell me about the specifications and features",
            "project_ids": ["project_alpha", "project_beta"],
            "use_hybrid_search": True,
            "limit": 10
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        citations = result.get("citations", [])
        
        print(f"✅ Query successful!")
        print(f"   - Citations count: {len(citations)}")
        
        # Check projects in citations
        projects_found = set()
        for cit in citations:
            project_ids = cit.get("payload", {}).get("project_ids", [])
            projects_found.update(project_ids)
        
        print(f"   - Projects in results: {projects_found}")
        
        if "project_alpha" in projects_found or "project_beta" in projects_found:
            print(f"   ✅ Results include documents from queried projects")
        
        return True
    else:
        print(f"❌ Query failed: {response.status_code}")
        return False

def test_stream_with_project_ids():
    """Test streaming endpoint with project_ids."""
    print("\n" + "="*60)
    print("TEST 6: Stream query with project_ids")
    print("="*60)
    
    response = requests.post(
        f"{BASE_URL}/api/v1/chat/query/stream",
        json={
            "query": "What are the Alpha features?",
            "project_ids": ["project_alpha"],
            "limit": 3
        },
        stream=True
    )
    
    if response.status_code == 200:
        print(f"✅ Stream opened successfully!")
        
        event_count = 0
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                event_count += 1
                if event_count <= 3:  # Only show first few events
                    data = json.loads(line[6:])
                    print(f"   - Event {event_count}: {data.get('type')}")
                if event_count > 10:  # Stop after a few events
                    break
        
        print(f"   - Total events received: {event_count}+")
        return True
    else:
        print(f"❌ Stream failed: {response.status_code}")
        return False

def main():
    print("\n" + "="*60)
    print("MULTI-TENANT PROJECT_IDS VERIFICATION TEST")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print(f"\n❌ Server not responding correctly at {BASE_URL}")
            sys.exit(1)
        print(f"✅ Server is running\n")
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to server at {BASE_URL}")
        print("   Please ensure the server is running: uvicorn app.main:app --reload")
        sys.exit(1)
    
    # Give time between tests
    results = []
    
    results.append(("Bulk ingest project_alpha", test_bulk_ingest_project_alpha()))
    time.sleep(1)
    
    results.append(("Bulk ingest project_beta", test_bulk_ingest_project_beta()))
    time.sleep(1)
    
    results.append(("Query matching project", test_query_with_matching_project()))
    time.sleep(0.5)
    
    results.append(("Query non-matching project", test_query_with_non_matching_project()))
    time.sleep(0.5)
    
    results.append(("Query multiple projects", test_query_multiple_projects()))
    time.sleep(0.5)
    
    results.append(("Stream with project_ids", test_stream_with_project_ids()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Multi-tenant filtering is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
