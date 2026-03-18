#!/bin/bash
# Setup Cron Jobs for PaperScope automated pipeline

# Get absolute path to the project root
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXEC="$PROJECT_DIR/venv/bin/python"

# Paths to the scripts
CRAWLER_SCRIPT="$PROJECT_DIR/crawler/pipelines/arxiv_scraper.py"
SUMMARIZER_SCRIPT="$PROJECT_DIR/summarizer/processor.py"
CLEANUP_SCRIPT="$PROJECT_DIR/backend/cleanup.py"

# Verify python venv exists
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: Python executable not found at $PYTHON_EXEC"
    echo "Please ensure the virtual environment 'venv' is created in the project root."
    exit 1
fi

echo "Setting up Cron jobs for PaperScope..."

# Create a temporary file for new crontab entries
TEMP_CRON=$(mktemp)

# Dump existing crontab (ignoring error if it doesn't exist)
crontab -l > "$TEMP_CRON" 2>/dev/null

# Clean up existing PaperScope entries to prevent duplicates
sed -i '/PaperScope/d' "$TEMP_CRON"

# Append new jobs
echo "" >> "$TEMP_CRON"
echo "## PaperScope Pipeline Automation ##" >> "$TEMP_CRON"

# 1. Run Scraper every day at 02:00 AM
echo "0 2 * * * cd $PROJECT_DIR && $PYTHON_EXEC $CRAWLER_SCRIPT >> $PROJECT_DIR/data/crawler.log 2>&1" >> "$TEMP_CRON"

# 2. Run AI Summarizer every day at 03:00 AM
echo "0 3 * * * cd $PROJECT_DIR && $PYTHON_EXEC $SUMMARIZER_SCRIPT >> $PROJECT_DIR/data/summarizer.log 2>&1" >> "$TEMP_CRON"

# 3. Clean up DB (older than 5 days) every day at 04:00 AM. (할당된 '5'를 바꿔 유지기간 조절 가능)
echo "0 4 * * * cd $PROJECT_DIR && $PYTHON_EXEC $CLEANUP_SCRIPT 5 >> $PROJECT_DIR/data/cleanup.log 2>&1" >> "$TEMP_CRON"

# Install the new crontab
crontab "$TEMP_CRON"
rm "$TEMP_CRON"

echo "Cron jobs installed successfully."
echo "Current crontab:"
crontab -l | grep -A 4 "PaperScope Pipeline"
