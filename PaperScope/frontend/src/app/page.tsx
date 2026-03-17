import { fetchPapers } from "@/api/papers";
import PaperCard from "@/components/PaperCard";

export default async function Home() {
  const papers = await fetchPapers(0, 50);

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black text-white p-8 md:p-12 lg:p-24">
      
      <div className="max-w-6xl mx-auto">
        <header className="mb-16 text-center space-y-4">
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-600">
            PaperScope
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Your personalized AI research radar. Tracking the latest papers tailored to your keywords.
          </p>
        </header>

        {papers.length === 0 ? (
          <div className="text-center p-12 bg-white/5 rounded-2xl border border-white/10">
            <p className="text-gray-400 text-lg">No papers found. Crawling pipeline might be empty.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {papers.map((paper) => (
              <PaperCard key={paper.id} paper={paper} />
            ))}
          </div>
        )}
      </div>
      
    </main>
  );
}
