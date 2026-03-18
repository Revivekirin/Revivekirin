import sys
import os
from datetime import datetime, timedelta

# Add the project root to sys.path so we can import backend properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import SessionLocal
from backend.models.paper import Paper

def delete_old_papers(days: int = 5):
    """
    Deletes papers from the database that were published more than `days` ago.
    """
    db = SessionLocal()
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        print(f"Cleaning up papers older than: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Find old papers
        old_papers = db.query(Paper).filter(Paper.published_date < cutoff_date)
        count = old_papers.count()
        
        if count > 0:
            old_papers.delete(synchronize_session=False)
            db.commit()
            print(f"Successfully deleted {count} old paper(s) from the database.")
        else:
            print("No old papers found. Database is clean.")
            
    except Exception as e:
        print(f"Error during cleanup: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    # 기본값은 5일이지만, 커맨드라인 인자(sys.argv)로 넘겨받을 수 있습니다.
    # 사용법: python cleanup.py 3 (3일 지난 데이터 삭제)
    days_to_keep = 5
    if len(sys.argv) > 1:
        try:
            days_to_keep = int(sys.argv[1])
        except ValueError:
            print("Invalid argument for days. Using default 5.")
            
    print(f"Starting DB cleanup for papers older than {days_to_keep} days...")
    delete_old_papers(days=days_to_keep)
