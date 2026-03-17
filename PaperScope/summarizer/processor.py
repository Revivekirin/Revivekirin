import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import SessionLocal
from backend.models.paper import Paper
from summarizer.models.openai_client import PaperSummarizer

def process_unsummarized_papers():
    """
    Finds papers in the database that haven't been summarized yet (or have the dummy summary)
    and processes them using the LLM.
    """
    db = SessionLocal()
    summarizer = PaperSummarizer()
    
    try:
        # Find papers with dummy summary or empty summary
        target_papers = db.query(Paper).filter(
            (Paper.ai_summary == "[Dummy Summary: Awaiting LLM Processing]") | 
            (Paper.ai_summary == None) | 
            (Paper.ai_summary == "")
        ).all()
        
        print(f"Found {len(target_papers)} papers needing summarization.")
        
        processed_count = 0
        for paper in target_papers:
            print(f"Summarizing: {paper.title[:50]}...")
            
            # Use OpenAI to generate summary
            summary = summarizer.summarize(title=paper.title, abstract=paper.abstract)
            
            # Update DB
            paper.ai_summary = summary
            db.commit()
            processed_count += 1
            print("  -> Success")
            
        print(f"Summarization complete. Processed {processed_count} papers.")
        
    except Exception as e:
         print(f"Fatal error during processing: {e}")
         db.rollback()
    finally:
         db.close()

if __name__ == "__main__":
    process_unsummarized_papers()
