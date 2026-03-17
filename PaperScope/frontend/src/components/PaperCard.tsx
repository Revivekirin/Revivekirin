import { Paper } from "@/api/papers";
import React from "react";

export default function PaperCard({ paper }: { paper: Paper }) {
  // Format the date
  const publishDate = new Date(paper.published_date).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  return (
    <div className="bg-white/5 backdrop-blur-xl border border-white/10 p-6 rounded-2xl shadow-xl hover:shadow-2xl hover:bg-white/10 transition-all duration-300 group">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-xl font-bold text-gray-100 group-hover:text-blue-400 transition-colors">
          <a href={paper.url} target="_blank" rel="noopener noreferrer">
            {paper.title}
          </a>
        </h3>
        <span className="px-3 py-1 text-xs font-semibold bg-blue-500/20 text-blue-300 rounded-full border border-blue-500/30 whitespace-nowrap ml-4">
          {paper.source}
        </span>
      </div>
      
      <p className="text-sm text-gray-400 mb-4 line-clamp-1">{paper.authors}</p>
      
      <div className="space-y-4">
        {/* AI Summary Badge and content */}
        <div className="p-4 bg-indigo-900/20 rounded-xl border border-indigo-500/20">
          <h4 className="flex items-center text-indigo-400 text-sm font-semibold mb-2">
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            AI Summary
          </h4>
          <p className="text-sm text-gray-300">
            {paper.ai_summary || "Summary pending..."}
          </p>
        </div>
      </div>

      <div className="mt-6 flex justify-between items-center border-t border-white/5 pt-4">
        <div className="flex flex-wrap gap-2">
          {paper.matched_keywords?.split(",").map((kw) => (
            <span key={kw} className="text-xs px-2 py-1 bg-gray-800 text-gray-300 rounded-md border border-gray-700">
              {kw.trim()}
            </span>
          ))}
        </div>
        <span className="text-xs text-gray-500">{publishDate}</span>
      </div>
    </div>
  );
}
