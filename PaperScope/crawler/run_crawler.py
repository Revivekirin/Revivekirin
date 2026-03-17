import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="PaperScope Crawler")
    parser.add_argument("--source", type=str, help="Source to crawl (e.g., arxiv, ieee)")
    args = parser.parse_args()

    print(f"Starting crawler for source: {args.source}")
    # Initialize pipeline
    # Fetch data
    # Parse data
    # Save to db or intermediate format
    print("Crawling complete.")

if __name__ == "__main__":
    main()
