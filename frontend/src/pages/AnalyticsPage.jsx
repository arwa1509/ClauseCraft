import React, { useState, useEffect } from 'react';
import { BarChart, PieChart, Info, ShieldCheck, Database, Filter, Search, ChevronDown } from 'lucide-react';
import axios from 'axios';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

const AnalyticsPage = () => {
  const [stats, setStats] = useState(null);
  const [entities, setEntities] = useState({});
  const [loading, setLoading] = useState(true);

  const API_BASE = 'http://localhost:8000/api';

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const statsResp = await axios.get(`${API_BASE}/retrieval/stats`);
      const entityResp = await axios.get(`${API_BASE}/ner/index/entities`);
      setStats(statsResp.data);
      setEntities(entityResp.data);
    } catch (err) {
      console.error('Failed to fetch analytics data', err);
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

  if (loading) return (
    <div className="flex items-center justify-center p-20">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-brand-600"></div>
    </div>
  );

  return (
    <div className="space-y-8 animate-fade-in">
      <div>
        <h1 className="text-3xl font-bold text-legal-900">System Analytics</h1>
        <p className="text-legal-500 mt-1">Insights into the legal document corpus and extracted knowledge graph.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card p-6 border-l-4 border-l-brand-600">
          <div className="flex items-center justify-between text-legal-500 mb-2">
            <span className="text-xs font-bold uppercase tracking-wider">Vector Index</span>
            <Database className="h-4 w-4" />
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-3xl font-bold text-legal-900">{stats?.vector_store_size || 0}</span>
            <span className="text-sm text-legal-500 font-medium">Chunks</span>
          </div>
          <p className="text-[10px] text-legal-400 mt-2">Densely embedded with all-MiniLM-L6-v2</p>
        </div>

        <div className="card p-6 border-l-4 border-l-amber-600">
          <div className="flex items-center justify-between text-legal-500 mb-2">
            <span className="text-xs font-bold uppercase tracking-wider">Entity Index</span>
            <Filter className="h-4 w-4" />
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-3xl font-bold text-legal-900">{stats?.entity_index_stats?.total_entities || 0}</span>
            <span className="text-sm text-legal-500 font-medium">Nodes</span>
          </div>
          <p className="text-[10px] text-legal-400 mt-2">Unique legal entities extracted</p>
        </div>

        <div className="card p-6 border-l-4 border-l-emerald-600">
          <div className="flex items-center justify-between text-legal-500 mb-2">
            <span className="text-xs font-bold uppercase tracking-wider">Relationships</span>
            <BarChart className="h-4 w-4" />
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-3xl font-bold text-legal-900">{stats?.entity_index_stats?.total_references || 0}</span>
            <span className="text-sm text-legal-500 font-medium">Edges</span>
          </div>
          <p className="text-[10px] text-legal-400 mt-2">Entity-to-chunk mappings</p>
        </div>

        <div className="card p-6 border-l-4 border-l-brand-900">
          <div className="flex items-center justify-between text-legal-500 mb-2">
            <span className="text-xs font-bold uppercase tracking-wider">System Integrity</span>
            <ShieldCheck className="h-4 w-4" />
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-2xl font-bold text-legal-900">Production</span>
          </div>
          <p className="text-[10px] text-legal-400 mt-2">NER + Fusion-Ranked Pipeline</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1 space-y-6">
          <div className="card">
            <div className="px-6 py-4 border-b border-legal-200 flex items-center justify-between bg-legal-50/50">
              <h3 className="font-semibold text-legal-900">Entity Distribution</h3>
              <PieChart className="h-4 w-4 text-legal-400" />
            </div>
            <div className="p-6 space-y-4">
               {Object.entries(stats?.entity_index_stats?.entities_by_label || {}).map(([label, count]) => (
                  <div key={label} className="space-y-1.5">
                    <div className="flex justify-between text-xs">
                      <span className="font-medium text-legal-700">{label}</span>
                      <span className="text-legal-500">{count}</span>
                    </div>
                    <div className="w-full bg-legal-100 rounded-full h-1.5">
                      <div 
                        className={cn("h-1.5 rounded-full", label === 'STATUTE' ? 'bg-blue-500' : 'bg-brand-600')} 
                        style={{ width: `${(count / (stats?.entity_index_stats?.total_entities || 1)) * 100}%` }}
                      ></div>
                    </div>
                  </div>
               ))}
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 card">
          <div className="px-6 py-4 border-b border-legal-200 flex items-center justify-between bg-legal-50/50">
            <h3 className="font-semibold text-legal-900">Knowledge Explorer</h3>
            <div className="flex items-center space-x-2">
               <div className="relative">
                 <input type="text" placeholder="Search entities..." className="pl-8 pr-4 py-1.5 text-xs border border-legal-200 rounded-lg focus:ring-1 focus:ring-brand-500 focus:border-brand-500 outline-none" />
                 <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-legal-400" />
               </div>
               <button className="p-1.5 border border-legal-200 rounded-lg text-legal-500 hover:bg-legal-100">
                  <Filter className="h-3.5 w-3.5" />
               </button>
            </div>
          </div>
          
          <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-8 content-start h-[500px] overflow-y-auto">
             {Object.entries(entities).map(([label, list]) => (
                <div key={label} className="space-y-4">
                   <h4 className="text-xs font-bold text-legal-400 uppercase tracking-widest flex items-center">
                     <span className={cn("w-2 h-2 rounded-full mr-2", label === 'STATUTE' ? 'bg-blue-500' : 'bg-brand-600')}></span>
                     {label}
                   </h4>
                   <div className="flex flex-wrap gap-2">
                     {list.slice(0, 15).map((item, idx) => (
                        <span key={idx} className={cn("entity-tag", getEntityStyle(label))}>
                          {item}
                        </span>
                     ))}
                     {list.length > 15 && <span className="text-xs text-legal-400 px-2 py-0.5">+{list.length - 15} more</span>}
                   </div>
                </div>
             ))}
             {Object.keys(entities).length === 0 && (
               <div className="col-span-2 py-20 text-center text-legal-400">
                 <Info className="h-10 w-10 mx-auto mb-4 opacity-20" />
                 <p>Process your documents to see extracted legal entities and knowledge distribution.</p>
               </div>
             )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsPage;
