import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class PaperSummarizer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
             print("WARNING: Valid GEMINI_API_KEY not found. Please set it in .env or environment variables.")
        else:
             genai.configure(api_key=self.api_key)
             
        # Using Gemini 1.5 Flash - generous free tier (1500 req/day)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def summarize(self, title: str, abstract: str) -> str:
        """Generates a summary for the given paper abstract using Google Gemini API."""
        if not abstract or len(abstract.strip()) < 10:
            return "Abstract too short to summarize."
            
        prompt = f"""You are an expert AI research assistant. Analyze the scientific paper abstract and provide a highly concise summary.
Focus strictly on: 1. Core problem. 2. Proposed method. 3. Key results.
Keep the summary under 3-4 sentences. Start immediately with the summary, no intro.

Title: {title}
Abstract: {abstract}"""
            
        try:
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            return summary
        except Exception as e:
            err_msg = str(e)
            print(f"Error generating summary: {err_msg}")
            return f"Error: Could not generate AI summary. ({err_msg})"
