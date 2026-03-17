from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

class PaperSummarizer:
    def __init__(self, api_key: str = None):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key or key == "your_openai_api_key_here":
             print("WARNING: Valid OPENAI_API_KEY not found. Summarizer will fast-fail or use dummy mode if allowed.")
             
        # Initialize the LLM (Using gpt-3.5-turbo or gpt-4o-mini for cost efficiency on summarization)
        self.llm = ChatOpenAI(api_key=key, model="gpt-4o-mini", temperature=0.3)
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI research assistant. Your task is to analyze scientific paper abstracts and provide a highly concise, easy-to-understand summary.
Focus on:
1. The core problem being solved.
2. The proposed method/architecture.
3. The key results or takeaways.

Keep the summary under 3-4 sentences. Format it nicely."""),
            ("user", "Title: {title}\n\nAbstract: {abstract}")
        ])
        
        # Create the chain
        self.chain = self.prompt | self.llm | StrOutputParser()

    def summarize(self, title: str, abstract: str) -> str:
        """Generates a summary for the given paper abstract."""
        if not abstract or len(abstract.strip()) < 10:
            return "Abstract too short to summarize."
            
        try:
            summary = self.chain.invoke({"title": title, "abstract": abstract})
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Error: Could not generate AI summary. ({str(e)})"
