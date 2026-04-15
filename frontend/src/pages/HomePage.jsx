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

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-4">
          <div className="card">
            <div className="px-6 py-4 border-b border-legal-200 flex items-center justify-between bg-legal-50/50">
              <h3 className="font-semibold text-legal-900">Document Queue</h3>
              <span className="text-xs font-medium text-legal-500">{docs.length} Documents</span>
            </div>
            <div className="divide-y divide-legal-100 max-h-[500px] overflow-y-auto">
              {docs.length === 0 ? (
                <div className="px-6 py-12 text-center">
                  <FileText className="h-12 w-12 text-legal-200 mx-auto mb-4" />
                  <p className="text-legal-500">No documents uploaded yet.</p>
                </div>
              ) : (
                docs.map((doc) => (
                  <div key={doc.name} className="px-6 py-4 flex items-center justify-between hover:bg-legal-50 transition-colors">
                    <div className="flex items-center space-x-3">
                      <div className="bg-legal-100 p-2 rounded-lg">
                        <FileText className="h-5 w-5 text-legal-600" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-legal-900">{doc.name}</p>
                        <p className="text-xs text-legal-500">{(doc.size / 1024).toFixed(1)} KB • {doc.type.toUpperCase()}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                       {/* Action buttons could go here */}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="card p-6 bg-brand-900 text-white border-0">
            <div className="flex items-center space-x-3 mb-4">
              <Database className="h-6 w-6 text-brand-300" />
              <h3 className="font-bold text-lg">System Metrics</h3>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between items-center py-2 border-b border-brand-800">
                <span className="text-brand-300 text-sm">Indexed Chunks</span>
                <span className="font-mono text-xl">0</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-brand-800">
                <span className="text-brand-300 text-sm">Legal Entities</span>
                <span className="font-mono text-xl">0</span>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="text-brand-300 text-sm">Embedding Model</span>
                <span className="text-xs bg-brand-800 px-2 py-1 rounded">MiniLM-L6-v2</span>
              </div>
            </div>
          </div>

          <div className="card p-6">
            <h3 className="font-bold text-legal-900 mb-4 flex items-center">
              <AlertCircle className="h-5 w-5 text-brand-600 mr-2" />
              Instructions
            </h3>
            <ul className="text-sm text-legal-600 space-y-3">
              <li className="flex items-start">
                <span className="bg-brand-100 text-brand-700 h-5 w-5 rounded-full flex items-center justify-center text-xs font-bold mr-2 mt-0.5">1</span>
                Upload your legal documents (PDF, TXT, or JSON).
              </li>
              <li className="flex items-start">
                <span className="bg-brand-100 text-brand-700 h-5 w-5 rounded-full flex items-center justify-center text-xs font-bold mr-2 mt-0.5">2</span>
                Click <strong>"Process & Index"</strong> to run the NLP pipeline.
              </li>
              <li className="flex items-start">
                <span className="bg-brand-100 text-brand-700 h-5 w-5 rounded-full flex items-center justify-center text-xs font-bold mr-2 mt-0.5">3</span>
                Wait for chunking, NER extraction, and vector embedding to complete.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
