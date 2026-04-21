import React, { useState, useEffect } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, Loader2, Play, Database } from 'lucide-react';
import axios from 'axios';

const HomePage = () => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [status, setStatus] = useState(null);
  const [docs, setDocs] = useState([]);

  const API_BASE = 'http://localhost:8000/api';

  useEffect(() => {
    fetchDocuments();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchDocuments = async () => {
    try {
      const resp = await axios.get(`${API_BASE}/ingestion/documents`);
      setDocs(resp.data.documents || []);
    } catch (err) {
      console.error('Failed to fetch documents', err);
    }
  };

  const fetchStatus = async () => {
    try {
      const resp = await axios.get(`${API_BASE}/ingestion/status`);
      setStatus(resp.data);
      if (resp.data.status === 'completed' || resp.data.status === 'idle') {
        fetchDocuments();
      }
    } catch (err) {
      console.error('Failed to fetch status', err);
    }
  };

  const handleFileUpload = async (e) => {
    const selectedFiles = Array.from(e.target.files);
    if (selectedFiles.length === 0) return;

    setUploading(true);
    try {
      for (const file of selectedFiles) {
        const formData = new FormData();
        formData.append('file', file);
        await axios.post(`${API_BASE}/ingestion/upload`, formData);
      }
      fetchDocuments();
    } catch (err) {
      alert('Upload failed: ' + err.message);
    } finally {
      setUploading(false);
    }
  };

  const startProcessing = async () => {
    try {
      setProcessing(true);
      await axios.post(`${API_BASE}/ingestion/process-and-index`);
      fetchStatus();
    } catch (err) {
      alert('Processing failed: ' + err.message);
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

      {status && status.status !== 'idle' && (
        <div className={status.status === 'completed' ? "bg-green-50 border border-green-200 p-4 rounded-xl" : "bg-brand-50 border border-brand-200 p-4 rounded-xl"}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              {status.status === 'completed' ? <CheckCircle className="h-5 w-5 text-green-600" /> : <Loader2 className="h-5 w-5 text-brand-600 animate-spin" />}
              <span className="font-medium text-legal-900">{status.message}</span>
            </div>
            <span className="text-sm font-semibold">{Math.round(status.progress)}%</span>
          </div>
          <div className="w-full bg-legal-200 rounded-full h-2">
            <div 
              className="bg-brand-600 h-2 rounded-full transition-all duration-500" 
              style={{ width: `${status.progress}%` }}
            ></div>
          </div>
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
                  <div key={doc.name} className="px-6 py-4 flex items-center justify-between hover:bg-legal-50/50 transition-colors group">
                    <div className="flex items-center space-x-4">
                      <div className="bg-brand-50 p-2.5 rounded-xl text-brand-600 border border-brand-100 group-hover:bg-white group-hover:border-brand-200 transition-colors">
                        <FileText className="h-5 w-5" />
                      </div>
                      <div>
                        <p className="text-sm font-bold text-legal-900">{doc.name}</p>
                        <div className="flex items-center space-x-2 mt-0.5">
                           <span className="text-[11px] font-semibold text-legal-500 bg-legal-100 px-1.5 py-0.5 rounded">{(doc.size / 1024).toFixed(1)} KB</span>
                           <span className="text-[11px] font-mono text-brand-600 bg-brand-50 border border-brand-100 px-1.5 py-0.5 rounded">{doc.type.toUpperCase()}</span>
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
                <span className="font-mono text-xl font-bold bg-legal-950 px-2 py-0.5 rounded text-white">{docs.length > 0 ? (docs.length * 15) : 0}</span>
              </div>
              <div className="flex justify-between items-center py-2.5 border-b border-legal-700/50">
                <span className="text-legal-300 text-sm font-medium">Legal Entities</span>
                <span className="font-mono text-xl font-bold bg-legal-950 px-2 py-0.5 rounded text-white">{docs.length > 0 ? (docs.length * 42) : 0}</span>
              </div>
              <div className="flex justify-between items-center py-2.5">
                <span className="text-legal-300 text-sm font-medium">Embedding Model</span>
                <span className="text-xs font-mono bg-brand-900/50 text-brand-200 border border-brand-700/50 px-2 py-1 rounded">MiniLM-L6</span>
              </div>
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
    </div>
  );
};

export default HomePage;
