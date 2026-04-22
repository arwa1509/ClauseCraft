import React, { useState, useEffect } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, Loader2, Play, Database } from 'lucide-react';
import axios from 'axios';

const HomePage = () => {
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [status, setStatus] = useState(null);
  const [docs, setDocs] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [traces, setTraces] = useState([]);
  const [errorMessage, setErrorMessage] = useState('');

  const API_BASE = `${import.meta.env.VITE_API_BASE || 'http://localhost:8000'}/api`;

  useEffect(() => {
    fetchDocuments();
  }, []);

  useEffect(() => {
    if (status?.status !== 'processing') {
      return undefined;
    }

    const timeoutId = setTimeout(() => {
      fetchStatus();
    }, 3000);

    return () => clearTimeout(timeoutId);
  }, [status]);

  const fetchDocuments = async () => {
    try {
      setErrorMessage('');
      const [docsResp, metricsResp, tracesResp] = await Promise.all([
        axios.get(`${API_BASE}/ingestion/documents`),
        axios.get(`${API_BASE}/ingestion/metrics`),
        axios.get(`${API_BASE}/ingestion/traces`),
      ]);
      setDocs(docsResp.data.documents || []);
      setMetrics(metricsResp.data.metrics || null);
      setTraces(tracesResp.data.documents || []);
    } catch (err) {
      console.error('Failed to fetch documents', err);
      setErrorMessage('Failed to load corpus metrics from the backend.');
    }
  };

  const fetchStatus = async () => {
    try {
      setErrorMessage('');
      const [statusResp, tracesResp] = await Promise.all([
        axios.get(`${API_BASE}/ingestion/status`),
        axios.get(`${API_BASE}/ingestion/traces`),
      ]);
      setStatus(statusResp.data);
      if (statusResp.data.metrics) {
        setMetrics(statusResp.data.metrics);
      }
      setTraces(tracesResp.data.documents || []);
      if (statusResp.data.status === 'completed') {
        fetchDocuments();
      }
    } catch (err) {
      console.error('Failed to fetch status', err);
      setErrorMessage('Failed to fetch ingestion status.');
    }
  };

  const handleFileUpload = async (e) => {
    const selectedFiles = Array.from(e.target.files);
    if (selectedFiles.length === 0) return;

    setUploading(true);
    try {
      setErrorMessage('');
      for (const file of selectedFiles) {
        const formData = new FormData();
        formData.append('file', file);
        await axios.post(`${API_BASE}/ingestion/upload`, formData);
      }
      fetchDocuments();
    } catch (err) {
      setErrorMessage(`Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  const startProcessing = async () => {
    try {
      setProcessing(true);
      setErrorMessage('');
      await axios.post(`${API_BASE}/ingestion/process-and-index`);
      fetchStatus();
    } catch (err) {
      setErrorMessage(`Processing failed: ${err.message}`);
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="space-y-8 animate-fade-in">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-legal-900">Document Corpus</h1>
          <p className="text-legal-500 mt-1">Ingest and process legal documents for NER-aware RAG.</p>
        </div>
        
        <div className="flex items-center space-x-3">
          <label className="btn-secondary cursor-pointer">
            <Upload className="h-4 w-4 mr-2" />
            Upload PDF/TXT
            <input type="file" multiple className="hidden" onChange={handleFileUpload} accept=".pdf,.txt,.json" />
          </label>
          
          <button 
            onClick={startProcessing}
            disabled={docs.length === 0 || status?.status === 'processing'}
            className="btn-primary"
          >
            {status?.status === 'processing' ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Play className="h-4 w-4 mr-2" />
            )}
            Process & Index
          </button>
        </div>
      </div>

      {errorMessage && (
        <div className="bg-red-50 border border-red-200 p-4 rounded-xl text-red-800">
          {errorMessage}
        </div>
      )}

      {status && status.status !== 'idle' && (
        <div className={status.status === 'completed' ? "bg-green-50 border border-green-200 p-4 rounded-xl" : "bg-brand-50 border border-brand-200 p-4 rounded-xl"}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              {status.status === 'completed' ? <CheckCircle className="h-5 w-5 text-green-600" /> : <Loader2 className="h-5 w-5 text-brand-600 animate-spin" />}
              <span className="font-medium text-legal-900">{status.message}</span>
            </div>
            <span className="text-sm font-semibold">{Math.round(status.progress)}%</span>
          </div>
          <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-legal-500">
            Stage: {status.current_stage || 'idle'}
          </div>
          <div className="w-full bg-legal-200 rounded-full h-2" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(status.progress)}>
            <div 
              className="bg-brand-600 h-2 rounded-full transition-all duration-500" 
              style={{ width: `${status.progress}%` }}
            ></div>
          </div>
          {status.documents?.length > 0 && (
            <div className="mt-4 grid gap-2">
              {status.documents.map((doc) => (
                <div key={`${doc.name}-${doc.status}`} className="flex items-center justify-between rounded-lg bg-white/70 px-3 py-2 text-sm border border-legal-100">
                  <span className="font-medium text-legal-800">{doc.name}</span>
                  <span className="text-legal-500">{doc.chunks} chunks</span>
                  <span className={`font-semibold ${doc.status === 'success' ? 'text-green-700' : 'text-red-700'}`}>{doc.status}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 pb-10">
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-white border border-legal-200 rounded-xl shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-legal-200 flex items-center justify-between bg-legal-50/50">
              <div className="flex items-center space-x-2">
                <FileText className="h-5 w-5 text-legal-700" />
                <h3 className="font-semibold text-legal-900">Document Queue</h3>
              </div>
              <span className="text-xs font-semibold bg-brand-100 text-brand-700 px-2 py-1 rounded-full">{docs.length} Uploaded</span>
            </div>
            <div className="divide-y divide-legal-100 max-h-[500px] overflow-y-auto styled-scrollbar">
              {docs.length === 0 ? (
                <div className="px-6 py-16 text-center">
                  <div className="bg-legal-50 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                    <FileText className="h-8 w-8 text-legal-300" />
                  </div>
                  <p className="text-legal-500 font-medium">No documents uploaded yet.</p>
                  <p className="text-legal-400 text-sm mt-1">Upload a PDF or TXT to begin pipeline ingestion.</p>
                </div>
              ) : (
                docs.map((doc) => (
                  <div key={`${doc.name}-${doc.modified}`} className="px-6 py-4 flex items-center justify-between hover:bg-legal-50/50 transition-colors group">
                    <div className="flex items-center space-x-4">
                      <div className="bg-brand-50 p-2.5 rounded-xl text-brand-600 border border-brand-100 group-hover:bg-white group-hover:border-brand-200 transition-colors">
                        <FileText className="h-5 w-5" />
                      </div>
                      <div>
                        <p className="text-sm font-bold text-legal-900">{doc.name}</p>
                        <div className="flex items-center space-x-2 mt-0.5">
                           <span className="text-[11px] font-semibold text-legal-500 bg-legal-100 px-1.5 py-0.5 rounded">{(doc.size / 1024).toFixed(1)} KB</span>
                           <span className="text-[11px] font-mono text-brand-600 bg-brand-50 border border-brand-100 px-1.5 py-0.5 rounded">{(doc.type || 'N/A').toUpperCase()}</span>
                        </div>
                      </div>
                    </div>
                    <div>
                        <CheckCircle className="h-5 w-5 text-green-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-gradient-to-br from-legal-900 to-legal-800 rounded-xl p-6 text-white border border-legal-700 shadow-md">
            <div className="flex items-center space-x-3 mb-5">
              <Database className="h-6 w-6 text-brand-400" />
              <h3 className="font-bold text-lg">System Metrics</h3>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between items-center py-2.5 border-b border-legal-700/50">
                <span className="text-legal-300 text-sm font-medium">Indexed Chunks</span>
                <span className="font-mono text-xl font-bold bg-legal-950 px-2 py-0.5 rounded text-white">{metrics?.chunks_total ?? 0}</span>
              </div>
              <div className="flex justify-between items-center py-2.5 border-b border-legal-700/50">
                <span className="text-legal-300 text-sm font-medium">Legal Entities</span>
                <span className="font-mono text-xl font-bold bg-legal-950 px-2 py-0.5 rounded text-white">{metrics?.entity_total ?? 0}</span>
              </div>
              <div className="flex justify-between items-center py-2.5 border-b border-legal-700/50">
                <span className="text-legal-300 text-sm font-medium">Page Citations</span>
                <span className="font-mono text-xl font-bold bg-legal-950 px-2 py-0.5 rounded text-white">{metrics?.page_citations_total ?? 0}</span>
              </div>
              <div className="flex justify-between items-center py-2.5">
                <span className="text-legal-300 text-sm font-medium">Embedding Model</span>
                <span className="text-xs font-mono bg-brand-900/50 text-brand-200 border border-brand-700/50 px-2 py-1 rounded">MiniLM-L6</span>
              </div>
            </div>
          </div>

          <div className="bg-white border border-legal-200 rounded-xl p-6 shadow-sm">
            <h3 className="font-bold text-legal-900 mb-4">Index Detail</h3>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="rounded-lg border border-legal-100 bg-legal-50 p-3">
                <p className="text-legal-500">Documents</p>
                <p className="mt-1 text-xl font-bold text-legal-900">{metrics?.documents_total ?? docs.length}</p>
              </div>
              <div className="rounded-lg border border-legal-100 bg-legal-50 p-3">
                <p className="text-legal-500">Entity References</p>
                <p className="mt-1 text-xl font-bold text-legal-900">{metrics?.entity_references_total ?? 0}</p>
              </div>
              <div className="rounded-lg border border-legal-100 bg-legal-50 p-3">
                <p className="text-legal-500">Avg Chunk Size</p>
                <p className="mt-1 text-xl font-bold text-legal-900">{metrics?.average_chunk_chars ?? 0}</p>
              </div>
              <div className="rounded-lg border border-legal-100 bg-legal-50 p-3">
                <p className="text-legal-500">Max Chunk Size</p>
                <p className="mt-1 text-xl font-bold text-legal-900">{metrics?.max_chunk_chars ?? 0}</p>
              </div>
            </div>

            <div className="mt-5 rounded-lg border border-legal-100 p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-legal-700">Vector Index</span>
                <span className={`text-xs font-semibold px-2 py-1 rounded-full ${metrics?.vector_index?.present ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                  {metrics?.vector_index?.present ? 'Ready' : 'Pending'}
                </span>
              </div>
              <p className="mt-2 text-sm text-legal-500">
                {metrics?.vector_index?.chunk_vectors ?? 0} chunk vectors available for retrieval.
              </p>
            </div>
          </div>

          <div className="bg-white border border-legal-200 rounded-xl p-6 shadow-sm">
            <h3 className="font-bold text-legal-900 mb-4">Corpus Breakdown</h3>
            <div className="space-y-3 text-sm">
              {Object.entries(metrics?.documents_by_type || {}).length === 0 ? (
                <p className="text-legal-500">No uploaded documents yet.</p>
              ) : (
                Object.entries(metrics?.documents_by_type || {}).map(([type, count]) => (
                  <div key={type} className="flex items-center justify-between">
                    <span className="uppercase font-semibold text-legal-600">{type}</span>
                    <span className="font-mono text-legal-900">{count}</span>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="bg-orange-50 border border-orange-100 rounded-xl p-6 shadow-sm">
            <h3 className="font-bold text-orange-900 mb-4 flex items-center">
              <AlertCircle className="h-5 w-5 text-orange-600 mr-2" />
              Ingestion Guide
            </h3>
            <ul className="text-sm text-orange-800 space-y-4">
              <li className="flex items-start">
                <span className="bg-orange-200 text-orange-800 h-5 w-5 rounded-full flex shrink-0 items-center justify-center text-xs font-bold mr-3 mt-0.5">1</span>
                <span>Upload your verified legal documents (PDF, TXT, or JSON).</span>
              </li>
              <li className="flex items-start">
                <span className="bg-orange-200 text-orange-800 h-5 w-5 rounded-full flex shrink-0 items-center justify-center text-xs font-bold mr-3 mt-0.5">2</span>
                <span>Click <strong>"Process & Index"</strong> to kick off the NLP parsing.</span>
              </li>
              <li className="flex items-start">
                <span className="bg-orange-200 text-orange-800 h-5 w-5 rounded-full flex shrink-0 items-center justify-center text-xs font-bold mr-3 mt-0.5">3</span>
                <span>Wait for semantic chunking, Named Entity Recognition mapping, and FAISS indexing.</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-white border border-legal-200 rounded-xl shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-legal-200 bg-legal-50/50">
          <h3 className="font-semibold text-legal-900">Per-Document Output</h3>
          <p className="text-sm text-legal-500 mt-1">Detailed ingestion results for each uploaded file.</p>
        </div>
        {metrics?.per_document?.length ? (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-legal-50 text-legal-600">
                <tr>
                  <th className="px-6 py-3 text-left font-semibold">Document</th>
                  <th className="px-6 py-3 text-left font-semibold">Type</th>
                  <th className="px-6 py-3 text-left font-semibold">Size</th>
                  <th className="px-6 py-3 text-left font-semibold">Chunks</th>
                  <th className="px-6 py-3 text-left font-semibold">Sections</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-legal-100">
                {metrics.per_document.map((doc) => (
                  <tr key={doc.name}>
                    <td className="px-6 py-4 font-medium text-legal-900">{doc.name}</td>
                    <td className="px-6 py-4 uppercase text-legal-600">{doc.type || 'n/a'}</td>
                    <td className="px-6 py-4 text-legal-600">{(doc.size_bytes / 1024).toFixed(1)} KB</td>
                    <td className="px-6 py-4 font-mono text-legal-900">{doc.chunks}</td>
                    <td className="px-6 py-4 font-mono text-legal-900">{doc.sections}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="px-6 py-10 text-sm text-legal-500">Upload and process documents to see detailed corpus output here.</div>
        )}
      </div>

      <div className="bg-white border border-legal-200 rounded-xl shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-legal-200 bg-legal-50/50">
          <h3 className="font-semibold text-legal-900">Parser And NER Trace</h3>
          <p className="text-sm text-legal-500 mt-1">How each document was interpreted from source text to chunks and named entities.</p>
        </div>
        {traces.length ? (
          <div className="divide-y divide-legal-100">
            {traces.map((trace) => (
              <div key={trace.document_name || trace.error} className="p-6 space-y-5">
                <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                  <div>
                    <h4 className="text-lg font-bold text-legal-900">{trace.document_name || 'Unknown document'}</h4>
                    <p className="text-sm text-legal-500 mt-1">
                      Parser: <span className="font-semibold text-legal-700">{trace.parser || 'unknown'}</span>
                      {' '}• Type: <span className="font-semibold text-legal-700">{trace.document_type || 'n/a'}</span>
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm md:min-w-[320px]">
                    <div className="rounded-lg bg-legal-50 border border-legal-100 p-3">
                      <p className="text-legal-500">Pages</p>
                      <p className="font-bold text-legal-900">{trace.page_count ?? 0}</p>
                    </div>
                    <div className="rounded-lg bg-legal-50 border border-legal-100 p-3">
                      <p className="text-legal-500">Chunks</p>
                      <p className="font-bold text-legal-900">{trace.chunk_count ?? 0}</p>
                    </div>
                    <div className="rounded-lg bg-legal-50 border border-legal-100 p-3">
                      <p className="text-legal-500">Raw Chars</p>
                      <p className="font-bold text-legal-900">{trace.raw_char_count ?? 0}</p>
                    </div>
                    <div className="rounded-lg bg-legal-50 border border-legal-100 p-3">
                      <p className="text-legal-500">Clean Chars</p>
                      <p className="font-bold text-legal-900">{trace.clean_char_count ?? 0}</p>
                    </div>
                  </div>
                </div>

                {trace.error ? (
                  <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
                    {trace.error}
                  </div>
                ) : (
                  <>
                    <div className="grid gap-4 lg:grid-cols-2">
                      <div className="rounded-xl border border-legal-100 p-4">
                        <h5 className="font-semibold text-legal-900">Raw Extraction Preview</h5>
                        <p className="mt-2 text-sm text-legal-600 leading-relaxed">{trace.raw_text_preview || 'No raw preview available.'}</p>
                      </div>
                      <div className="rounded-xl border border-legal-100 p-4">
                        <h5 className="font-semibold text-legal-900">Cleaned Text Preview</h5>
                        <p className="mt-2 text-sm text-legal-600 leading-relaxed">{trace.clean_text_preview || 'No cleaned preview available.'}</p>
                      </div>
                    </div>

                    <div className="grid gap-4 lg:grid-cols-2">
                      <div className="rounded-xl border border-legal-100 p-4">
                        <h5 className="font-semibold text-legal-900">Detected Sections</h5>
                        <div className="mt-3 flex flex-wrap gap-2">
                          {trace.detected_sections?.length ? trace.detected_sections.map((section) => (
                            <span key={section} className="rounded-full bg-brand-50 border border-brand-100 px-2.5 py-1 text-xs font-semibold text-brand-700">
                              {section}
                            </span>
                          )) : (
                            <span className="text-sm text-legal-500">No explicit section headings detected.</span>
                          )}
                        </div>
                      </div>
                      <div className="rounded-xl border border-legal-100 p-4">
                        <h5 className="font-semibold text-legal-900">NER Label Distribution</h5>
                        <div className="mt-3 flex flex-wrap gap-2">
                          {Object.entries(trace.entity_labels || {}).length ? Object.entries(trace.entity_labels).map(([label, count]) => (
                            <span key={label} className="rounded-full bg-legal-50 border border-legal-100 px-2.5 py-1 text-xs font-semibold text-legal-700">
                              {label}: {count}
                            </span>
                          )) : (
                            <span className="text-sm text-legal-500">Entity labels appear here after indexing finishes.</span>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="grid gap-4 lg:grid-cols-2">
                      <div className="rounded-xl border border-legal-100 p-4">
                        <h5 className="font-semibold text-legal-900">Page Interpretation</h5>
                        <div className="mt-3 space-y-3 max-h-80 overflow-y-auto styled-scrollbar pr-1">
                          {trace.page_samples?.length ? trace.page_samples.map((page) => (
                            <div key={`${trace.document_name}-page-${page.page_num}`} className="rounded-lg bg-legal-50 border border-legal-100 p-3">
                              <div className="flex items-center justify-between text-xs font-semibold text-legal-600">
                                <span>Page {page.page_num}</span>
                                <span>{page.char_count} chars{page.has_tables ? ' • tables' : ''}</span>
                              </div>
                              {page.sections?.length > 0 && (
                                <div className="mt-2 flex flex-wrap gap-1">
                                  {page.sections.slice(0, 4).map((section) => (
                                    <span key={section} className="rounded-full bg-white border border-legal-200 px-2 py-0.5 text-[11px] text-legal-600">
                                      {section}
                                    </span>
                                  ))}
                                </div>
                              )}
                              <p className="mt-2 text-sm text-legal-600 leading-relaxed">{page.preview}</p>
                            </div>
                          )) : (
                            <p className="text-sm text-legal-500">No page-level breakdown available for this file type.</p>
                          )}
                        </div>
                      </div>

                      <div className="rounded-xl border border-legal-100 p-4">
                        <h5 className="font-semibold text-legal-900">Chunking And Entities</h5>
                        <div className="mt-3 space-y-3 max-h-80 overflow-y-auto styled-scrollbar pr-1">
                          {trace.chunk_samples?.length ? trace.chunk_samples.map((chunk) => (
                            <div key={chunk.chunk_id || `${trace.document_name}-${chunk.page_num}`} className="rounded-lg bg-legal-50 border border-legal-100 p-3">
                              <div className="flex items-center justify-between text-xs font-semibold text-legal-600">
                                <span>{chunk.chunk_id || 'chunk'}</span>
                                <span>{chunk.char_count} chars</span>
                              </div>
                              <div className="mt-1 text-[11px] text-legal-500">
                                Page {chunk.page_num ?? 'n/a'}{chunk.section ? ` • ${chunk.section}` : ''}{chunk.section_num ? ` ${chunk.section_num}` : ''}
                              </div>
                              <p className="mt-2 text-sm text-legal-600 leading-relaxed">{chunk.preview}</p>
                            </div>
                          )) : (
                            <p className="text-sm text-legal-500">Chunk samples appear here after parsing completes.</p>
                          )}
                        </div>
                        <div className="mt-4">
                          <h6 className="text-sm font-semibold text-legal-800">Sample Entities</h6>
                          <div className="mt-2 flex flex-wrap gap-2">
                            {trace.sample_entities?.length ? trace.sample_entities.map((entity, idx) => (
                              <span key={`${entity.text}-${idx}`} className="rounded-full bg-yellow-50 border border-yellow-100 px-2.5 py-1 text-xs font-semibold text-yellow-800">
                                {entity.text} ({entity.label})
                              </span>
                            )) : (
                              <span className="text-sm text-legal-500">Sample entities appear here after NER indexing.</span>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="px-6 py-10 text-sm text-legal-500">Process documents to inspect parser, chunking, and NER traces here.</div>
        )}
      </div>
    </div>
  );
};

export default HomePage;
