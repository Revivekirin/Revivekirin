from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import papers

app = FastAPI(title="PaperScope API", version="1.0.0")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to PaperScope API!"}

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# Include Routers
app.include_router(papers.router, prefix="/api")

