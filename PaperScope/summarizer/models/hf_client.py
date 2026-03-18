import os
import requests
from dotenv import load_dotenv

load_dotenv()

class PaperSummarizer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key or self.api_key == "your_hf_api_key_here":
             print("WARNING: Valid HUGGINGFACE_API_KEY not found. Hugging Face Inference API requires a free token.")
             
        # Using Qwen2.5-7B-Instruct which is successfully working on the huggingface router
        self.model_id = "Qwen/Qwen2.5-7B-Instruct"
        # The new official HF Inference Router API endpoint for chat models
        self.api_url = "https://router.huggingface.co/v1/chat/completions"

    def summarize(self, title: str, abstract: str) -> str:
        """Generates a summary for the given paper abstract via standard HTTP requests."""
        if not abstract or len(abstract.strip()) < 10:
            return "Abstract too short to summarize."
            
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Hugging Face Router API (v1/chat/completions) strictly requires the OpenAI Chat format
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "You are an expert AI research assistant. Analyze the scientific paper abstract and provide a highly concise summary.\nFocus strictly on: 1. Core problem. 2. Proposed method. 3. Key results.\nKeep the summary under 3-4 sentences. Start immediately with the summary, no intro."},
                {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"}
            ],
            "max_tokens": 150,
            "temperature": 0.3
        }
            
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract standard chat completions style response
            summary = result["choices"][0]["message"]["content"].strip()
            return summary
        except Exception as e:
            err_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                err_msg += f". Response: {e.response.text}"
            print(f"Error generating summary: {err_msg}")
            return f"Error: Could not generate AI summary. ({err_msg})"
