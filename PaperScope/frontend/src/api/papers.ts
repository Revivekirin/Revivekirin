export interface Paper {
    id: number;
    title: string;
    authors: string;
    source: string;
    published_date: string;
    url: string;
    abstract: string | null;
    ai_summary: string | null;
    matched_keywords: string | null;
  }
  
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";
  
  export async function fetchPapers(skip: number = 0, limit: number = 50): Promise<Paper[]> {
    try {
      const response = await fetch(`${API_BASE_URL}/papers/?skip=${skip}&limit=${limit}`, {
        // Fetch fresh data explicitly or rely on Next.js default cache
        cache: 'no-store', 
      });
      if (!response.ok) {
        throw new Error('Failed to fetch papers');
      }
      return response.json();
    } catch (error) {
      console.error("Error fetching papers:", error);
      return [];
    }
  }
