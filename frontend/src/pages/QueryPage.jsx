import React, { useState } from 'react';
import { Search, Send, Brain, Shield, Info, ChevronRight, ExternalLink, BarChart3, Highlighter } from 'lucide-react';
import axios from 'axios';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

const QueryPage = () => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('answer');

  const API_BASE = 'http://localhost:8000/api';

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const resp = await axios.post(`${API_BASE}/query`, {
        question: query,
        top_k: 5,
        use_entity_retrieval: true,
        use_reranking: true
      });
      setResult(resp.data);
    } catch (err) {
      console.error('Query failed', err);
      setError(err.response?.data?.detail || 'An error occurred while processing your request.');
    } finally {
      setLoading(false);
    }
  };

  const getEntityStyle = (label) => {
    const styles = {
      STATUTE: "bg-blue-100 text-blue-800 border-blue-200",
      PROVISION: "bg-purple-100 text-purple-800 border-purple-200",
      COURT: "bg-pink-100 text-pink-800 border-pink-200",
      JUDGE: "bg-amber-100 text-amber-800 border-amber-200",
      CASE_CITATION: "bg-emerald-100 text-emerald-800 border-emerald-200",
      LEGAL_ACTION: "bg-orange-100 text-orange-800 border-orange-200",
      DATE: "bg-indigo-100 text-indigo-800 border-indigo-200",
      PARTY: "bg-violet-100 text-violet-800 border-violet-200",
      ACT: "bg-cyan-100 text-cyan-800 border-cyan-200",
      PENALTY: "bg-red-100 text-red-800 border-red-200",
    };
    return styles[label] || "bg-gray-100 text-gray-800 border-gray-200";
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="text-center max-w-3xl mx-auto space-y-3">
        <h1 className="text-3xl font-bold text-legal-900 tracking-tight">Legal Knowledge Retrieval</h1>
        <p className="text-legal-600 text-base">Ask complex legal questions and get explainable, entity-aware answers grounded in your document corpus.</p>
        
        <form onSubmit={handleSearch} className="relative mt-6">
          <input
            type="text"
            className="w-full pl-10 pr-20 py-3 bg-white border border-legal-200 rounded-xl shadow-sm focus:ring-2 focus:ring-brand-500/20 focus:border-brand-600 transition-all text-base"
            placeholder="e.g., What is the punishment for theft under section 378 of IPC?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-legal-400" />
          <button
            type="submit"
            disabled={loading}
            className="absolute right-2 top-1/2 -translate-y-1/2 btn-primary px-4 py-1.5 rounded-lg text-sm"
          >
            {loading ? <span className="animate-pulse">Searching...</span> : <><Send className="h-4 w-4 mr-1.5" /> Ask</>}
          </button>
        </form>
      </div>

      {error && (
        <div className="max-w-4xl mx-auto bg-red-50 border border-red-200 p-3 rounded-lg flex items-center text-red-700 text-sm">
          <Shield className="h-4 w-4 mr-2 flex-shrink-0" />
          <p>{error}</p>
        </div>
      )}

      {loading && (
        <div className="max-w-4xl mx-auto space-y-4">
          <div className="card p-6 text-center space-y-3">
            <div className="flex justify-center">
              <div className="relative h-10 w-10">
                <div className="absolute inset-0 border-2 border-brand-100 rounded-full"></div>
                <div className="absolute inset-0 border-2 border-brand-600 rounded-full border-t-transparent animate-spin"></div>
              </div>
            </div>
            <p className="text-legal-600 text-sm font-medium animate-pulse">Running advanced legal NLP pipeline...</p>
            <div className="flex justify-center space-x-6 text-[10px] text-legal-400 uppercase tracking-wider">
              <span className="flex items-center"><ChevronRight className="h-3 w-3 mr-1" /> NER Mapping</span>
              <span className="flex items-center"><ChevronRight className="h-3 w-3 mr-1" /> Hybrid Retrieval</span>
              <span className="flex items-center"><ChevronRight className="h-3 w-3 mr-1" /> Cross-Ranking</span>
              <span className="flex items-center"><ChevronRight className="h-3 w-3 mr-1" /> Extraction</span>
            </div>
          </div>
        </div>
      )}

      {result && (
        <div className="max-w-5xl mx-auto space-y-6">
          {/* We now expect result.answer to be a JSON string describing the 4 sections */}
          {(() => {
            let extractedData = null;
            try {
              extractedData = JSON.parse(result.answer);
            } catch (e) {
              // Fallback just in case it's not JSON
              extractedData = {
                simple_answer: result.answer,
                supporting_text: "Could not parse supporting text.",
                key_entities: [],
                confidence: 0.0
              };
            }

            return (
              <div className="grid gap-4">
                {/* Section 1: Simple Answer */}
                <div className="card p-5 border-l-4 border-l-brand-600">
                  <div className="flex items-center space-x-2 mb-2 text-brand-700">
                    <Brain className="h-5 w-5" />
                    <h2 className="text-sm font-bold uppercase tracking-wider">Simple Answer</h2>
                  </div>
                  <p className="text-base text-legal-900 leading-relaxed">
                    {extractedData.simple_answer}
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Section 2: Supporting Text */}
                  <div className="card p-5">
                    <div className="flex items-center space-x-2 mb-3 text-legal-600">
                      <Highlighter className="h-4 w-4" />
                      <h3 className="text-base font-semibold">Supporting Text</h3>
                    </div>
                    <div className="bg-legal-50 p-3 rounded-lg text-sm text-legal-800 leading-relaxed h-40 overflow-y-auto border border-legal-100">
                      {extractedData.supporting_text}
                    </div>
                  </div>

                  <div className="space-y-4">
                    {/* Section 3: Key Entities */}
                    <div className="card p-5">
                      <div className="flex items-center space-x-2 mb-3 text-legal-600">
                        <Info className="h-4 w-4" />
                        <h3 className="text-base font-semibold">Key Entities</h3>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {extractedData.key_entities?.length > 0 ? (
                          extractedData.key_entities.map((ent, idx) => (
                            <span key={idx} className="bg-indigo-50 text-indigo-700 px-2.5 py-1 rounded text-xs font-semibold border border-indigo-100">
                              {ent}
                            </span>
                          ))
                        ) : (
                          <span className="text-sm text-legal-400">No entities found.</span>
                        )}
                      </div>
                    </div>

                    {/* Section 4: Confidence Score */}
                    <div className="card p-5">
                      <div className="flex items-center justify-between mb-3 text-legal-600">
                        <div className="flex items-center space-x-2">
                          <BarChart3 className="h-4 w-4" />
                          <h3 className="text-base font-semibold">Confidence Score</h3>
                        </div>
                        <span className="text-lg font-bold text-brand-600">
                          {Math.round(extractedData.confidence * 100)}%
                        </span>
                      </div>
                      <div className="w-full bg-legal-100 rounded-full h-2.5 overflow-hidden border border-legal-200 relative">
                        <div 
                          className="bg-brand-500 h-2.5 rounded-full transition-all duration-1000 ease-out absolute left-0 top-0" 
                          style={{ width: `${Math.max(0, Math.min(100, extractedData.confidence * 100))}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
};

export default QueryPage;
