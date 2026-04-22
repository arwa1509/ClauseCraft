import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Send, FileText, Bot, User, Loader2 } from "lucide-react";
import { cn } from "../utils/cn";

const WELCOME_MESSAGE = {
  id: "welcome",
  type: "bot",
  content: "Ask about the indexed legal documents and I will answer from supported passages only.",
};

const QueryPage = () => {
  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem("clausecraft-messages");
      return saved ? JSON.parse(saved) : [WELCOME_MESSAGE];
    } catch {
      return [WELCOME_MESSAGE];
    }
  });
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  useEffect(() => {
    try {
      localStorage.setItem("clausecraft-messages", JSON.stringify(messages));
    } catch {
      // ignore storage quota errors
    }
  }, [messages]);

  const API_BASE = `${import.meta.env.VITE_API_BASE || "http://localhost:8000"}/api`;

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userMessage = { id: crypto.randomUUID(), type: "user", content: query };
    setMessages((prev) => [...prev, userMessage]);
    setQuery("");
    setLoading(true);
    abortControllerRef.current = new AbortController();

    try {
      const resp = await axios.post(`${API_BASE}/query`, {
        question: userMessage.content,
        top_k: 5,
        use_entity_retrieval: true,
        use_reranking: true
      }, {
        signal: abortControllerRef.current.signal,
      });

      const answerData = resp.data.answer || {
        simple_answer: "No grounded answer was returned.",
        markdown_answer: "",
        answer_text: "",
        key_entities: [],
        confidence: 0,
        supporting_passages: [],
        citations: [],
        source: "local",
      };

      const topSupport = answerData.supporting_passages?.[0] || null;

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          type: "bot",
          content: answerData.markdown_answer || answerData.simple_answer,
          plain_answer: answerData.simple_answer,
          supporting_text: answerData.answer_text || topSupport?.text || "",
          key_entities: answerData.key_entities,
          confidence: typeof answerData.confidence === "number" ? answerData.confidence : 0,
          source_meta: topSupport?.metadata || {
            source: answerData.source || "local"
          },
          citations: answerData.citations || [],
          evidence_points: answerData.evidence_points || [],
          answer_segments: answerData.answer_segments || [],
          supporting_passages: answerData.supporting_passages || [],
          answer_type: answerData.answer_type || "grounded_extractive",
          web_augmented: Boolean(answerData.web_augmented),
          web_results_count: answerData.web_results_count || 0,
          web_error: answerData.web_error || null,
          explanation: resp.data.explanation || {},
          query_entities: resp.data.query_entities || [],
          answer_entities: resp.data.entities || [],
          retrieved_sources: resp.data.sources || [],
          overall_confidence: resp.data.confidence || 0,
          processing_time: resp.data.processing_time || 0,
        }
      ]);
    } catch (err) {
      console.error("Query failed", err);
      const content = axios.isCancel(err)
        ? "Query cancelled."
        : err.response?.data?.detail || "An error occurred while connecting to the legal knowledge base.";
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          type: "error",
          content
        }
      ]);
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  };

  const cancelSearch = () => {
    abortControllerRef.current?.abort();
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
        <p className="text-xs text-legal-500 mt-1 pl-8 font-medium">Grounded answers from indexed passages with visible source support.</p>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 bg-gradient-to-b from-gray-50 to-white/70 relative w-full styled-scrollbar" role="log" aria-live="polite">
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
                {msg.type === "bot" ? (
                  <div className="font-serif prose prose-sm max-w-none prose-p:my-1 prose-li:my-0.5">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  </div>
                ) : (
                  <p className="font-serif">{msg.content}</p>
                )}

                {/* Supporting Extra UI (only for bot responses with payload) */}
                {msg.supporting_text && (
                  <div className="mt-4 pt-4 border-t border-legal-100 border-dashed space-y-3">
                    
                    {msg.plain_answer && (
                      <div className="rounded-xl border border-brand-100 bg-brand-50/60 p-4 text-sm text-legal-800">
                        <p className="text-xs font-bold uppercase tracking-wide text-brand-700 mb-2">Short Answer</p>
                        <p className="font-serif leading-relaxed">{msg.plain_answer}</p>
                      </div>
                    )}

                    {msg.evidence_points?.length > 0 && (
                      <div className="rounded-xl border border-orange-100 bg-orange-50/50 p-4">
                        <p className="text-xs font-bold uppercase tracking-wide text-orange-700 mb-3">Evidence Chain</p>
                        <div className="space-y-3">
                          {msg.evidence_points.map((point, idx) => (
                            <div key={`${msg.id}-evidence-${idx}`} className="rounded-lg border border-orange-100 bg-white/70 p-3 text-sm text-legal-700">
                              <p className="font-serif leading-relaxed">{point.text}</p>
                              <div className="mt-2 text-[11px] text-legal-500">
                                {point.citation_ids?.map((id) => `[${id}]`).join(" ")} {point.section ? `• ${point.section}` : ""} {point.page_num ? `• p.${point.page_num}` : ""}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Context block */}
                    <div className="bg-orange-50/50 rounded-xl p-4 text-sm text-legal-700 font-serif border border-orange-100 shadow-inner relative group leading-relaxed">
                      <div className="absolute top-3 right-3 text-orange-300">
                        <FileText size={16} />
                      </div>
                      <p className="text-xs font-bold uppercase tracking-wide text-orange-700 mb-2">Answer Text Used</p>
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
                            {ent}
                          </span>
                        ))}
                        {msg.key_entities?.length > 3 && (
                          <span className="text-[10px] font-bold text-legal-400 px-1">+ {msg.key_entities.length - 3}</span>
                        )}
                      </div>

                      {/* Confidence */}
                      <div className={cn(
                        "flex items-center space-x-2 px-2 py-1 rounded border",
                        msg.confidence >= 0.75 ? "bg-green-50 border-green-100 text-green-700" :
                        msg.confidence >= 0.45 ? "bg-yellow-50 border-yellow-100 text-yellow-700" :
                        "bg-red-50 border-red-100 text-red-700"
                      )}>
                        <span className="text-[11px] font-bold uppercase tracking-wide">Support</span>
                        <div className={cn(
                          "w-12 h-1.5 rounded-full overflow-hidden",
                          msg.confidence >= 0.75 ? "bg-green-200" :
                          msg.confidence >= 0.45 ? "bg-yellow-200" :
                          "bg-red-200"
                        )}>
                          <div 
                            className={cn(
                              "h-full rounded-full",
                              msg.confidence >= 0.75 ? "bg-green-500" :
                              msg.confidence >= 0.45 ? "bg-yellow-500" :
                              "bg-red-500"
                            )}
                            style={{width: `${Math.max(0, Math.min(100, msg.confidence * 100))}%`}} 
                          />
                        </div>
                        <span className="text-[11px] font-bold">{Math.round(msg.confidence * 100)}%</span>
                      </div>

                    </div>
                    {msg.citations?.length > 0 && (
                      <div className="rounded-xl border border-legal-100 bg-legal-50/50 p-4">
                        <p className="text-xs font-bold uppercase tracking-wide text-legal-600 mb-3">Citations</p>
                        <div className="space-y-2 text-[12px] text-legal-600">
                          {msg.citations.map((citation) => (
                            <div key={`${msg.id}-${citation.id}`} className="rounded-lg border border-legal-100 bg-white px-3 py-2">
                              {citation.url ? (
                                <a href={citation.url} target="_blank" rel="noopener noreferrer" className="font-semibold text-brand-700 hover:underline">
                                  [{citation.id}] {citation.doc_name}
                                </a>
                              ) : (
                                <span className="font-semibold text-legal-800">[{citation.id}] {citation.doc_name}</span>
                              )}
                              <div className="mt-1 text-legal-500">
                                {citation.section ? `Section: ${citation.section} • ` : ""}
                                {citation.page_num ? `Page: ${citation.page_num} • ` : ""}
                                Source: {citation.source}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {msg.supporting_passages?.length > 0 && (
                      <div className="rounded-xl border border-legal-100 bg-legal-50/50 p-4">
                        <p className="text-xs font-bold uppercase tracking-wide text-legal-600 mb-3">Supporting Passages</p>
                        <div className="space-y-3">
                          {msg.supporting_passages.map((passage, idx) => (
                            <div key={`${msg.id}-passage-${passage.chunk_id || idx}`} className="rounded-lg border border-legal-100 bg-white p-3">
                              <div className="flex items-center justify-between text-[11px] text-legal-500 mb-2">
                                <span>{passage.metadata?.doc_name || "Local document"}</span>
                                <span>Score {Math.round((passage.score || 0) * 100)}%</span>
                              </div>
                              <p className="text-sm text-legal-700 leading-relaxed font-serif">{passage.text}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {(msg.explanation || msg.query_entities?.length || msg.answer_entities?.length) && (
                      <div className="rounded-xl border border-legal-100 bg-legal-50/50 p-4">
                        <p className="text-xs font-bold uppercase tracking-wide text-legal-600 mb-3">Pipeline Detail</p>
                        <div className="grid gap-4 lg:grid-cols-2 text-sm">
                          <div>
                            <p className="font-semibold text-legal-800">Retrieval</p>
                            <div className="mt-2 space-y-1 text-legal-600">
                              <p>Dense results: {msg.explanation?.num_dense_results ?? 0}</p>
                              <p>Entity results: {msg.explanation?.num_entity_results ?? 0}</p>
                              <p>Fused results: {msg.explanation?.num_fused_results ?? 0}</p>
                              <p>Reranked: {msg.explanation?.num_reranked ?? 0}</p>
                              <p>Overall confidence: {Math.round((msg.overall_confidence || 0) * 100)}%</p>
                              <p>Processing time: {msg.processing_time}s</p>
                            </div>
                          </div>
                          <div>
                            <p className="font-semibold text-legal-800">NER</p>
                            <div className="mt-2 space-y-2">
                              <div>
                                <p className="text-[11px] font-bold uppercase tracking-wide text-legal-500">Query Entities</p>
                                <div className="mt-1 flex flex-wrap gap-1.5">
                                  {msg.query_entities?.length ? msg.query_entities.map((entity, idx) => (
                                    <span key={`${msg.id}-qent-${idx}`} className="rounded-full bg-brand-50 border border-brand-100 px-2 py-0.5 text-[11px] font-semibold text-brand-700">
                                      {entity.text} ({entity.label})
                                    </span>
                                  )) : <span className="text-legal-500">No query entities detected.</span>}
                                </div>
                              </div>
                              <div>
                                <p className="text-[11px] font-bold uppercase tracking-wide text-legal-500">Answer Entities</p>
                                <div className="mt-1 flex flex-wrap gap-1.5">
                                  {msg.answer_entities?.length ? msg.answer_entities.slice(0, 10).map((entity, idx) => (
                                    <span key={`${msg.id}-aent-${idx}`} className="rounded-full bg-yellow-50 border border-yellow-100 px-2 py-0.5 text-[11px] font-semibold text-yellow-800">
                                      {entity.text} ({entity.label})
                                    </span>
                                  )) : <span className="text-legal-500">No answer entities detected.</span>}
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                        {msg.explanation?.claim_mapping?.length > 0 && (
                          <div className="mt-4">
                            <p className="font-semibold text-legal-800 mb-2">Claim Mapping</p>
                            <div className="space-y-2">
                              {msg.explanation.claim_mapping.map((mapping, idx) => (
                                <div key={`${msg.id}-claim-${idx}`} className="rounded-lg border border-legal-100 bg-white p-3 text-sm">
                                  <p className="font-medium text-legal-800">{mapping.claim}</p>
                                  <p className="mt-2 text-legal-600">{mapping.evidence_snippet || mapping.source_text}</p>
                                  <p className="mt-2 text-[11px] text-legal-500">
                                    Similarity {Math.round((mapping.similarity || 0) * 100)}%
                                    {mapping.source_metadata?.doc_name ? ` • ${mapping.source_metadata.doc_name}` : ""}
                                    {mapping.source_metadata?.page_num ? ` • p.${mapping.source_metadata.page_num}` : ""}
                                  </p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                    {msg.retrieved_sources?.length > 0 && (
                      <div className="rounded-xl border border-legal-100 bg-legal-50/50 p-4">
                        <p className="text-xs font-bold uppercase tracking-wide text-legal-600 mb-3">Retrieved Sources</p>
                        <div className="space-y-3">
                          {msg.retrieved_sources.map((source, idx) => (
                            <div key={`${msg.id}-source-${source.chunk_id || idx}`} className="rounded-lg border border-legal-100 bg-white p-3">
                              <div className="flex items-center justify-between text-[11px] text-legal-500 mb-2">
                                <span>{source.metadata?.doc_name || "Local document"}</span>
                                <span>Score {Math.round((source.score || 0) * 100)}%</span>
                              </div>
                              <p className="text-sm text-legal-700 leading-relaxed font-serif">{source.text}</p>
                              {source.entities?.length > 0 && (
                                <div className="mt-2 flex flex-wrap gap-1.5">
                                  {source.entities.slice(0, 8).map((entity, entityIdx) => (
                                    <span key={`${msg.id}-src-entity-${idx}-${entityIdx}`} className="rounded-full bg-legal-50 border border-legal-100 px-2 py-0.5 text-[11px] font-semibold text-legal-600">
                                      {entity.text} ({entity.label})
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {(msg.web_augmented || msg.web_error) && (
                      <div className="text-[11px] text-legal-500 pt-2">
                        {msg.web_augmented ? (
                          <span className="inline-flex items-center rounded-full bg-blue-50 px-2 py-1 font-semibold text-blue-700 border border-blue-100">
                            Web-augmented answer • {msg.web_results_count} external source{msg.web_results_count === 1 ? '' : 's'}
                          </span>
                        ) : (
                          <span className="inline-flex items-center rounded-full bg-legal-50 px-2 py-1 font-semibold text-legal-600 border border-legal-100">
                            Web fallback not used • {msg.web_error}
                          </span>
                        )}
                      </div>
                    )}
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
                <span className="text-sm text-legal-500 animate-pulse">Finding supported passages...</span>
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
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            aria-label="Send legal query"
            className="absolute right-2 top-1/2 -translate-y-1/2 bg-legal-900 hover:bg-legal-800 disabled:bg-legal-200 disabled:text-legal-400 text-white p-2.5 rounded-lg transition-colors"
          >
            <Send size={18} className={cn(query.trim() && !loading ? "translate-x-0.5" : "")} />
          </button>
        </form>
        {loading && (
          <div className="text-center mt-3">
            <button type="button" onClick={cancelSearch} className="text-xs font-medium text-brand-700 hover:text-brand-900 underline">
              Cancel query
            </button>
          </div>
        )}
        <div className="text-center mt-3">
          <p className="text-[11px] text-legal-400">ClauseCraft answers from indexed passages and shows source support. If support is weak, it says so.</p>
        </div>
      </div>

    </div>
  );
};

export default QueryPage;
