import sys
import os
import time
import re
from datetime import datetime

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import SessionLocal
from backend.models.paper import Paper
from summarizer.models.hf_client import PaperSummarizer


MIN_REQUEST_INTERVAL = 15  # free tier 5 RPM 기준 안전 여유
MAX_RETRIES_PER_PAPER = 6


def extract_retry_delay(error_message: str, default: int = 60) -> int:
    """
    Gemini 429 에러 메시지에서 'Please retry in 40.97s' 같은 값을 파싱.
    실패하면 default 반환.
    """
    if not error_message:
        return default

    match = re.search(r"retry in\s+([0-9.]+)s", error_message, re.IGNORECASE)
    if match:
        seconds = float(match.group(1))
        return max(int(seconds) + 1, 1)

    # retry_delay { seconds: 40 } 형태도 대비
    match = re.search(r"retry_delay\s*\{[^}]*seconds:\s*(\d+)", error_message, re.IGNORECASE)
    if match:
        return max(int(match.group(1)) + 1, 1)

    return default


def summarize_with_retry(summarizer, title: str, abstract: str) -> str:
    """
    429 quota 에러가 나면 자동으로 기다렸다가 재시도.
    그 외 에러는 바로 raise.
    """
    attempt = 0

    while attempt < MAX_RETRIES_PER_PAPER:
        try:
            return summarizer.summarize(title=title, abstract=abstract)

        except Exception as e:
            error_text = str(e)

            # 429 quota 초과면 대기 후 재시도
            if "429" in error_text or "quota" in error_text.lower() or "rate limit" in error_text.lower():
                wait_seconds = extract_retry_delay(error_text, default=60)
                attempt += 1
                print(
                    f"  -> Rate limit hit (attempt {attempt}/{MAX_RETRIES_PER_PAPER}). "
                    f"Waiting {wait_seconds}s before retry..."
                )
                time.sleep(wait_seconds)
                continue

            # quota 외 에러는 그대로 올림
            raise

    raise RuntimeError(f"Exceeded max retries for paper: {title[:80]}")


def process_unsummarized_papers():
    """
    Finds papers in the database that haven't been summarized yet
    and processes them using the LLM with retry/backoff.
    """
    db = SessionLocal()
    summarizer = PaperSummarizer()

    try:
        target_papers = db.query(Paper).filter(
            (Paper.ai_summary == "[Dummy Summary: Awaiting LLM Processing]") |
            (Paper.ai_summary == None) |
            (Paper.ai_summary == "")
        ).all()

        print(f"Found {len(target_papers)} papers needing summarization.")

        processed_count = 0
        failed_count = 0
        last_request_time = 0.0

        for idx, paper in enumerate(target_papers, start=1):
            print(f"[{idx}/{len(target_papers)}] Summarizing: {paper.title[:80]}")

            try:
                # 요청 간 최소 간격 보장
                elapsed = time.time() - last_request_time
                if elapsed < MIN_REQUEST_INTERVAL:
                    sleep_time = MIN_REQUEST_INTERVAL - elapsed
                    print(f"  -> Waiting {sleep_time:.1f}s to respect base rate limit...")
                    time.sleep(sleep_time)

                summary = summarize_with_retry(
                    summarizer=summarizer,
                    title=paper.title,
                    abstract=paper.abstract
                )
                last_request_time = time.time()

                paper.ai_summary = summary
                db.commit()
                processed_count += 1
                print("  -> Success.")

            except Exception as e:
                db.rollback()
                failed_count += 1
                print(f"  -> Failed: {e}")

                # 실패했더라도 다음 요청 전에 약간 쉬기
                time.sleep(10)

        print(
            f"Summarization complete. "
            f"Processed={processed_count}, Failed={failed_count}, Total={len(target_papers)}"
        )

    except Exception as e:
        print(f"Fatal error during processing: {e}")
        db.rollback()

    finally:
        db.close()


if __name__ == "__main__":
    process_unsummarized_papers()