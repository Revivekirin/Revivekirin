# PaperScope
An automated pipeline to crawl, summarize, and display research papers from top AI conferences and repositories.

## 🚀 Local Quickstart

### 1. Environment Setup
Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
pip install -r crawler/requirements.txt
pip install -r summarizer/requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory and add your free Hugging Face API key:
```env
# .env
HUGGINGFACE_API_KEY=hf_your_token_here
DATABASE_URL=sqlite:///./paperscope.db
```
*Note: You must grant the "Make calls to the serverless Inference API" permission to your Hugging Face token.*

You can configure search keywords in `crawler/keywords.json`.

### 3. Running the Pipeline Manually
Run the following commands in order to populate the database and clean up old data:
```bash
# Ensure virtual environment is active
source venv/bin/activate

# 1. Crawl new papers based on keywords
python crawler/pipelines/arxiv_scraper.py

# 2. Summarize abstracts using Hugging Face AI (Qwen2.5-7B)
python summarizer/processor.py

# 3. (Optional) Delete papers older than X days (e.g., 0 days clears all past papers)
python backend/cleanup.py 0
```

### 4. Start the Web Servers
Start the backend API and frontend in separate terminals:

**Backend (FastAPI):**
```bash
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend (Next.js):**
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view heavily summarized AI papers!

## Directory Structure

- `backend/`: FastAPI application server.
- `crawler/`: Daily paper crawling scripts.
- `summarizer/`: AI summarization modules.
- `frontend/`: React/Next.js web client.
- `database/`: Schema migrations and setup.
- `data/`: Temporary storage for raw and processed paper records.
- `deploy/`: Docker files and server configurations.
