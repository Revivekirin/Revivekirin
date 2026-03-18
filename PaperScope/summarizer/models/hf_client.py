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
             
        # Using Gemini 2.5 Flash from the supported models list
        # Fallback to gemini-flash-latest just in case
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.fallback_model = genai.GenerativeModel('gemini-flash-latest')

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
            try:
                response = self.model.generate_content(prompt)
            except Exception as e:
                # If 1.5-flash-latest fails (e.g. 404), fallback to gemini-pro
                print(f"Fallback to gemini-pro due to: {e}")
                response = self.fallback_model.generate_content(prompt)
                
            summary = response.text.strip()
            return summary
        except Exception as e:
            err_msg = str(e)
            print(f"Error generating summary: {err_msg}")
            return f"Error: Could not generate AI summary. ({err_msg})"
