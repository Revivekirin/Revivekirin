import arxiv
import sys
import os
import json
from datetime import datetime

# Add the project root to sys.path so we can import backend properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.database import SessionLocal
from backend.models.paper import Paper

def extract_conference(summary: str, comment: str):
    """
    Very crude method to extract top tier AI conference mentions from arXiv metadata.
    A more robust approach would use regex or NLP on the abstract/comments.
    """
    targets = ["NeurIPS", "ICML", "ICLR", "CVPR", "AAAI", "ACL", "EMNLP"]
    text = (summary + " " + (comment or "")).upper()
    
    for t in targets:
        if t.upper() in text:
            return t
    return "arXiv"

def scrape_arxiv(keywords: list, max_results: int = 20):
    client = arxiv.Client()
    
    # Construct a query string dynamically from keywords
    query_parts = []
    for kw in keywords:
        # Wrap exact phrases in quotes
        query_parts.append(f'all:"{kw}"')
    
    query = " OR ".join(query_parts)
    print(f"Scraping arXiv with query: {query}")
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    db = SessionLocal()
    added_count = 0
    
    try:
        results = list(client.results(search))
        print(f"Found {len(results)} papers from fetch.")
        
        for r in results:
            # Check if paper already exists
            exists = db.query(Paper).filter(Paper.url == r.entry_id).first()
            if exists:
                continue
                
            conf = extract_conference(r.summary, r.comment)
            authors_str = ", ".join([a.name for a in r.authors])
            
            # Identify which keyword matched (simple substring check for display)
            matched = []
            lower_title_abs = (r.title + " " + r.summary).lower()
            for kw in keywords:
                if kw.lower() in lower_title_abs:
                    matched.append(kw)
                    
            paper_obj = Paper(
                title=r.title,
                authors=authors_str,
                source=conf, # e.g. arXiv, or NeurIPS if guessed
                published_date=r.published,
                url=r.entry_id,
                abstract=r.summary,
                ai_summary="[Dummy Summary: Awaiting LLM Processing]", # Fake summary
                matched_keywords=", ".join(matched) if matched else "General",
            )
            
            db.add(paper_obj)
            added_count += 1
            
        db.commit()
        print(f"Successfully added {added_count} new papers to the database.")
        
    except Exception as e:
        print(f"Error during scraping: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    # Load keywords from centralized config
    config_path = os.path.join(os.path.dirname(__file__), "..", "keywords.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            TARGET_KEYWORDS = config.get("target_keywords", [])
            max_results = config.get("scraper_limits", {}).get("arxiv", 20)
    except Exception as e:
        print(f"Failed to load keywords config: {e}")
        sys.exit(1)
        
    if not TARGET_KEYWORDS:
        print("No target keywords found in config. Exiting.")
        sys.exit(0)
    
    print(f"Starting arXiv scraper for keywords: {TARGET_KEYWORDS}")
    scrape_arxiv(TARGET_KEYWORDS, max_results=max_results)
