import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { Send, FileText, Bot, User, Loader2 } from "lucide-react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

const QueryPage = () => {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      type: "bot",
      content: "Hello. I am LexAnalyze, your Extractive Legal AI. What legal queries can I help you extract from your corpus today?",
    }
  ]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const API_BASE = "http://localhost:8000/api";

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = { id: Date.now(), type: "user", content: query };
    setMessages((prev) => [...prev, userMessage]);
    setQuery("");
    setLoading(true);

    try {
      const resp = await axios.post(`${API_BASE}/query`, {
        question: userMessage.content,
        top_k: 5,
        use_entity_retrieval: true,
        use_reranking: true
      });

      // Safely parse the backend JSON extraction
      let answerData;
      try {
        answerData = JSON.parse(resp.data.answer);
      } catch (err) {
        answerData = {
          simple_answer: resp.data.answer,
          supporting_text: "No supporting text structured properly.",
          key_entities: [],
          confidence: 0,
          source_meta: null
        };
      }

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          type: "bot",
          content: answerData.simple_answer,
          supporting_text: answerData.supporting_text,
          key_entities: answerData.key_entities,
          confidence: answerData.confidence,
          source_meta: answerData.source_meta
        }
      ]);
    } catch (err) {
      console.error("Query failed", err);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          type: "error",
          content: err.response?.data?.detail || "An error occurred while connecting to the legal knowledge base."
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full w-full max-w-5xl mx-auto rounded-xl overflow-hidden border border-legal-200 bg-white shadow-xl bg-opacity-95 backdrop-blur-sm relative">
      
      {/* Header */}
      <div className="bg-legal-100/50 border-b border-legal-200 px-6 py-4 flex flex-col shrink-0 flex-none z-10 backdrop-blur-xl">
        <div className="flex items-center space-x-2">
           <div className="bg-brand-600 p-1.5 rounded text-white flex items-center justify-center shadow-sm">
             <Bot size={16}/>
           </div>
           <h1 className="text-xl font-extrabold text-legal-950 tracking-tight">Legal Co-Pilot</h1>
        </div>
        <p className="text-xs text-legal-500 mt-1 pl-8 font-medium">Deterministic Extraction & Semantic Verification.</p>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 bg-gradient-to-b from-gray-50 to-white/70 relative w-full styled-scrollbar">
        {messages.map((msg) => (
          <div key={msg.id} className={cn("flex", msg.type === "user" ? "justify-end" : "justify-start")}>
            <div className={cn(
              "flex space-x-3 max-w-[85%]",
              msg.type === "user" ? "flex-row-reverse space-x-reverse" : "flex-row"
            )}>
              
              {/* Avatar */}
              <div className={cn(
                "flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center border",
                msg.type === "user" ? "bg-legal-100 border-legal-300 text-legal-700" : 
                msg.type === "error" ? "bg-red-100 border-red-300 text-red-600" :
                "bg-brand-100 border-brand-300 text-brand-700"
              )}>
                {msg.type === "user" ? <User size={16} /> : <Bot size={16} />}
              </div>

              {/* Message Bubble */}
              <div className={cn(
                "rounded-2xl px-5 py-3.5 text-[15px] leading-relaxed shadow-sm",
                msg.type === "user" ? "bg-legal-900 text-white rounded-tr-sm" : 
                msg.type === "error" ? "bg-red-50 text-red-900 border border-red-200 rounded-tl-sm" :
                "bg-white border border-legal-200 text-legal-800 rounded-tl-sm"
              )}>
                {/* Main Content */}
                <p className="font-serif">{msg.content}</p>

                {/* Supporting Extra UI (only for bot responses with payload) */}
                {msg.supporting_text && (
                  <div className="mt-4 pt-4 border-t border-legal-100 border-dashed space-y-3">
                    
                    {/* Context block */}
                    <div className="bg-orange-50/50 rounded-xl p-4 text-sm text-legal-700 font-serif border border-orange-100 shadow-inner relative group leading-relaxed">
                      <div className="absolute top-3 right-3 text-orange-300">
                        <FileText size={16} />
                      </div>
                      <p className="pr-6 italic font-medium">"...{msg.supporting_text}..."</p>
                    </div>

                    {/* Meta info row */}
                    <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
                      
                      {/* Meta Source & Link */}
                      {msg.source_meta && (
                        <div className="flex items-center space-x-1.5 flex-1 bg-yellow-50/50 border border-yellow-100 rounded px-2 py-1">
                          <FileText size={12} className="text-yellow-600" />
                          {msg.source_meta.url ? (
                             <a href={msg.source_meta.url} target="_blank" rel="noopener noreferrer" className="text-[11px] font-semibold text-yellow-700 hover:underline truncate max-w-[200px]">
                               {msg.source_meta.title || "External Source"}
                             </a>
                          ) : (
                             <span className="text-[11px] font-semibold text-yellow-700 truncate max-w-[200px]">
                               {msg.source_meta.doc_name || msg.source_meta.source || "Local Document"}
                             </span>
                          )}
                        </div>
                      )}

                      {/* Entities */}
                      <div className="flex flex-wrap gap-1.5 flex-1">
                        {msg.key_entities?.slice(0, 3).map((ent, idx) => (
                          <span key={idx} className="bg-brand-50 text-brand-700 text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded shadow-sm border border-brand-100">
                            {ent.split(" ")[0]}
                          </span>
                        ))}
                        {msg.key_entities?.length > 3 && (
                          <span className="text-[10px] font-bold text-legal-400 px-1">+ {msg.key_entities.length - 3}</span>
                        )}
                      </div>

                      {/* Confidence */}
                      <div className="flex items-center space-x-2 bg-green-50 border border-green-100 px-2 py-1 rounded text-green-700">
                        <span className="text-[11px] font-bold uppercase tracking-wide">Match</span>
                        <div className="w-12 bg-green-200 h-1.5 rounded-full overflow-hidden">
                          <div 
                            className="bg-green-500 h-full rounded-full" 
                            style={{width: `${Math.max(0, Math.min(100, msg.confidence * 100))}%`}} 
                          />
                        </div>
                        <span className="text-[11px] font-bold">{Math.round(msg.confidence * 100)}%</span>
                      </div>

                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="flex space-x-3">
              <div className="flex-shrink-0 h-8 w-8 rounded-full bg-brand-100 border border-brand-300 flex items-center justify-center text-brand-700">
                <Bot size={16} />
              </div>
              <div className="bg-white border border-legal-200 rounded-2xl rounded-tl-sm px-5 py-4 shadow-sm flex items-center space-x-2">
                <Loader2 className="w-4 h-4 text-brand-500 animate-spin" />
                <span className="text-sm text-legal-500 animate-pulse">Running NLTK extraction...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 md:p-6 bg-white border-t border-legal-200 shrink-0 shadow-[0_-15px_30px_-15px_rgba(0,0,0,0.05)] z-10 w-full flex-none">
        <form onSubmit={handleSearch} className="relative max-w-4xl mx-auto flex items-center">
          <input
            type="text"
            className="w-full pl-6 pr-16 py-4 bg-legal-50/80 border border-legal-200 rounded-full focus:ring-4 focus:ring-brand-500/10 focus:border-brand-400 transition-all text-[15px] outline-none shadow-inner text-legal-900 placeholder:text-legal-400"
            placeholder="Ask a legal query... (e.g. What is the penalty under Section 302 IPC?)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="absolute right-2 top-1/2 -translate-y-1/2 bg-legal-900 hover:bg-legal-800 disabled:bg-legal-200 disabled:text-legal-400 text-white p-2.5 rounded-lg transition-colors"
          >
            <Send size={18} className={cn(query.trim() && !loading ? "translate-x-0.5" : "")} />
          </button>
        </form>
        <div className="text-center mt-3">
          <p className="text-[11px] text-legal-400">LexAnalyze AI handles your specific queries using deterministic NLP, eliminating hallucination risks.</p>
        </div>
      </div>

    </div>
  );
};

export default QueryPage;

