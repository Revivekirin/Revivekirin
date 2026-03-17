# PaperScope

PaperScope is an automated pipeline that crawls the latest research papers based on specified keywords, summarizes them using AI, and serves the results on a modern web application.

## Directory Structure

- `backend/`: FastAPI application server.
- `crawler/`: Daily paper crawling scripts.
- `summarizer/`: AI summarization modules.
- `frontend/`: React/Next.js web client.
- `database/`: Schema migrations and setup.
- `data/`: Temporary storage for raw and processed paper records.
- `deploy/`: Docker files and server configurations.
