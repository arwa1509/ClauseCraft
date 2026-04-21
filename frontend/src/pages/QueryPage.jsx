import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';

// ─── Pixel/Retro Design Tokens (inline so no extra CSS file needed) ──────────
const RETRO = {
  bg: '#0a0a0f',
  panel: '#0f0f1a',
  panelBorder: '#1a1a2e',
  neonGreen: '#39ff14',
  neonMagenta: '#ff00ff',
  neonCyan: '#00ffff',
  neonYellow: '#ffe600',
  dimGreen: '#1a4d00',
  dimMagenta: '#4d004d',
  dimCyan: '#003333',
  textPrimary: '#e0ffe0',
  textMuted: '#4d7a4d',
  textDim: '#2a3d2a',
  scanline: 'rgba(0,255,0,0.03)',
};

// ─── Pixel border CSS (8-bit corner effect via box-shadow) ───────────────────
const pixelBorderStyle = (color = RETRO.neonGreen, size = 2) => ({
  border: `${size}px solid ${color}`,
  boxShadow: `
    0 0 0 ${size}px ${RETRO.bg},
    0 0 0 ${size * 2}px ${color},
    0 0 8px ${color}44,
    inset 0 0 8px ${color}11
  `,
});

const glowText = (color = RETRO.neonGreen) => ({
  color,
  textShadow: `0 0 8px ${color}, 0 0 16px ${color}88`,
});

// ─── Scanline overlay ────────────────────────────────────────────────────────
const ScanlineOverlay = () => (
  <div style={{
    position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 9999,
    backgroundImage: `repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      ${RETRO.scanline} 2px,
      ${RETRO.scanline} 4px
    )`,
  }} />
);

// ─── Pixel corner decorations ─────────────────────────────────────────────────
const PixelCorners = ({ color = RETRO.neonGreen }) => {
  const sz = 10;
  const c = { position: 'absolute', width: sz, height: sz, background: color };
  return (
    <>
      <div style={{ ...c, top: 0, left: 0 }} />
      <div style={{ ...c, top: 0, right: 0 }} />
      <div style={{ ...c, bottom: 0, left: 0 }} />
      <div style={{ ...c, bottom: 0, right: 0 }} />
    </>
  );
};

// ─── Flicker animation hook ───────────────────────────────────────────────────
function useFlicker() {
  const [opacity, setOpacity] = useState(1);
  useEffect(() => {
    const flicker = () => {
      const rand = Math.random();
      setOpacity(rand > 0.97 ? 0.85 : 1);
    };
    const id = setInterval(flicker, 80);
    return () => clearInterval(id);
  }, []);
  return opacity;
}

// ─── Typewriter component ─────────────────────────────────────────────────────
function TypewriterText({ text, speed = 18, color = RETRO.neonGreen }) {
  const [displayed, setDisplayed] = useState('');
  const idx = useRef(0);

  useEffect(() => {
    setDisplayed('');
    idx.current = 0;
    const id = setInterval(() => {
      if (idx.current < text.length) {
        setDisplayed(prev => prev + text[idx.current]);
        idx.current++;
      } else {
        clearInterval(id);
      }
    }, speed);
    return () => clearInterval(id);
  }, [text, speed]);

  return (
    <span style={{ color, fontFamily: "'Share Tech Mono', 'Courier New', monospace" }}>
      {displayed}
      <span style={{ animation: 'blink 1s step-end infinite', color }}>▋</span>
    </span>
  );
}

// ─── Entity badge ─────────────────────────────────────────────────────────────
const ENTITY_COLORS = {
  STATUTE:      { fg: RETRO.neonCyan,    bg: RETRO.dimCyan },
  PROVISION:    { fg: RETRO.neonMagenta, bg: RETRO.dimMagenta },
  COURT:        { fg: '#ff9900',         bg: '#331f00' },
  JUDGE:        { fg: '#ffff00',         bg: '#333300' },
  CASE_CITATION:{ fg: RETRO.neonGreen,   bg: RETRO.dimGreen },
  LEGAL_ACTION: { fg: '#ff4444',         bg: '#330000' },
  DATE:         { fg: '#aa88ff',         bg: '#220033' },
  PARTY:        { fg: '#00ccff',         bg: '#002233' },
  ACT:          { fg: '#33ffcc',         bg: '#003322' },
  PENALTY:      { fg: '#ff6666',         bg: '#330011' },
};

function EntityBadge({ label, text }) {
  const { fg, bg } = ENTITY_COLORS[label] || { fg: RETRO.textMuted, bg: '#111' };
  const [hovered, setHovered] = useState(false);

  return (
    <span
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'inline-block',
        background: hovered ? fg : bg,
        color: hovered ? RETRO.bg : fg,
        border: `1px solid ${fg}`,
        boxShadow: hovered ? `0 0 8px ${fg}` : `0 0 3px ${fg}44`,
        borderRadius: 2,
        padding: '2px 8px',
        fontSize: 11,
        fontFamily: "'Share Tech Mono', monospace",
        cursor: 'default',
        transition: 'all 0.15s',
        margin: '2px',
        letterSpacing: '0.05em',
        textTransform: 'uppercase',
      }}
      title={`Type: ${label}`}
    >
      [{label}] {text}
    </span>
  );
}

// ─── Confidence gauge (pixel progress bar) ───────────────────────────────────
function ConfidenceGauge({ score, webAugmented }) {
  const pct = Math.round(score * 100);
  const barColor = pct >= 75 ? RETRO.neonGreen : pct >= 50 ? RETRO.neonYellow : RETRO.neonMagenta;
  const blocks = Math.round(pct / 5); // 20 blocks total

  return (
    <div style={{ fontFamily: "'Share Tech Mono', monospace" }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, alignItems: 'center' }}>
        <span style={{ ...glowText(barColor), fontSize: 12, letterSpacing: '0.1em' }}>
          CONFIDENCE SCORE
        </span>
        {webAugmented && (
          <span style={{
            fontSize: 10, padding: '2px 6px',
            background: RETRO.dimMagenta,
            color: RETRO.neonMagenta,
            border: `1px solid ${RETRO.neonMagenta}`,
            boxShadow: `0 0 6px ${RETRO.neonMagenta}66`,
            letterSpacing: '0.1em',
          }}>
            WEB AUGMENTED
          </span>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        {/* Pixel blocks */}
        <div style={{ display: 'flex', gap: 2 }}>
          {Array.from({ length: 20 }, (_, i) => (
            <div
              key={i}
              style={{
                width: 12, height: 20,
                background: i < blocks ? barColor : '#1a1a1a',
                boxShadow: i < blocks ? `0 0 4px ${barColor}` : 'none',
                border: `1px solid ${i < blocks ? barColor + '44' : '#333'}`,
                transition: `background 0.05s ${i * 25}ms`,
              }}
            />
          ))}
        </div>
        <span style={{ ...glowText(barColor), fontSize: 22, fontWeight: 'bold' }}>
          {pct}%
        </span>
      </div>
    </div>
  );
}

// ─── Citation chunk card ──────────────────────────────────────────────────────
function CitationCard({ chunk, index, isHighlighted, answer }) {
  const [expanded, setExpanded] = useState(false);
  const text = chunk?.text || '';
  const meta = chunk?.metadata || {};
  const source = meta.doc_name || meta.source_file || `Chunk ${index + 1}`;
  const pageNum = meta.page_num || chunk?.page_num;

  // Highlight query-relevant words in the chunk (naive matching against answer tokens)
  const highlight = useCallback((rawText, referenceText) => {
    if (!referenceText || !rawText) return rawText;
    const tokens = new Set(
      referenceText.toLowerCase().split(/\W+/).filter(w => w.length > 4)
    );
    const re = new RegExp(`\\b(${[...tokens].map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})\\b`, 'gi');
    const parts = rawText.split(re);
    return parts.map((part, i) =>
      tokens.has(part.toLowerCase())
        ? <mark key={i} style={{ background: RETRO.dimGreen, color: RETRO.neonGreen, padding: '0 2px', borderRadius: 2 }}>{part}</mark>
        : part
    );
  }, []);

  return (
    <div
      style={{
        position: 'relative',
        background: isHighlighted ? '#0d1f0d' : RETRO.panel,
        padding: 14,
        marginBottom: 10,
        cursor: 'pointer',
        transition: 'all 0.2s',
        ...pixelBorderStyle(isHighlighted ? RETRO.neonGreen : RETRO.panelBorder),
      }}
      onClick={() => setExpanded(e => !e)}
    >
      <PixelCorners color={isHighlighted ? RETRO.neonGreen : '#1a1a2e'} />

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
        <div>
          <span style={{ ...glowText(RETRO.neonCyan), fontSize: 11, letterSpacing: '0.15em' }}>
            ◈ SOURCE [{index + 1}]
          </span>
          <span style={{ color: RETRO.textMuted, fontSize: 11, marginLeft: 10, fontFamily: 'monospace' }}>
            {source}{pageNum ? ` · p.${pageNum}` : ''}
          </span>
        </div>
        <span style={{ color: RETRO.textMuted, fontSize: 11, fontFamily: 'monospace' }}>
          {expanded ? '▲ COLLAPSE' : '▼ EXPAND'}
        </span>
      </div>

      <p style={{
        color: RETRO.textPrimary,
        fontSize: 12,
        lineHeight: 1.7,
        fontFamily: "'Share Tech Mono', monospace",
        margin: 0,
        maxHeight: expanded ? 'none' : 60,
        overflow: 'hidden',
        transition: 'max-height 0.3s',
      }}>
        {highlight(text.slice(0, expanded ? undefined : 200), answer)}
        {!expanded && text.length > 200 && '…'}
      </p>

      {chunk?.entities?.length > 0 && expanded && (
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {chunk.entities.map((ent, ei) => (
            <EntityBadge key={ei} label={ent.label} text={ent.text} />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── JSON Inspector Terminal ──────────────────────────────────────────────────
function JsonInspector({ data }) {
  const [open, setOpen] = useState(false);

  return (
    <div style={{ marginTop: 12 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          background: 'none',
          border: `1px solid ${RETRO.neonMagenta}`,
          color: RETRO.neonMagenta,
          fontFamily: "'Share Tech Mono', monospace",
          fontSize: 11,
          padding: '4px 12px',
          cursor: 'pointer',
          letterSpacing: '0.1em',
          boxShadow: open ? `0 0 8px ${RETRO.neonMagenta}` : 'none',
          transition: 'box-shadow 0.2s',
        }}
      >
        {open ? '◼ HIDE RAW JSON' : '◻ INSPECT RAW JSON'}
      </button>

      {open && (
        <div style={{
          marginTop: 8,
          padding: 16,
          background: '#070712',
          ...pixelBorderStyle(RETRO.neonMagenta, 1),
          overflowX: 'auto',
          maxHeight: 400,
          overflowY: 'auto',
        }}>
          <pre style={{
            margin: 0,
            fontFamily: "'Share Tech Mono', 'Courier New', monospace",
            fontSize: 11,
            color: RETRO.neonMagenta,
            whiteSpace: 'pre-wrap',
            textShadow: `0 0 4px ${RETRO.neonMagenta}88`,
          }}>
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

// ─── Loading terminal effect ──────────────────────────────────────────────────
const PIPELINE_STEPS = [
  '> Initialising NER entity mapper...',
  '> Running FAISS semantic search (768-dim)...',
  '> Entity index lookup (Rule ML hybrid)...',
  '> Applying Reciprocal Rank Fusion...',
  '> Evaluating confidence threshold...',
  '> Checking agentic fallback router...',
  '> Generating grounded answer...',
];

function LoadingTerminal() {
  const [stepIdx, setStepIdx] = useState(0);

  useEffect(() => {
    if (stepIdx >= PIPELINE_STEPS.length) return;
    const id = setTimeout(() => setStepIdx(i => i + 1), 380);
    return () => clearTimeout(id);
  }, [stepIdx]);

  return (
    <div style={{
      padding: 20,
      background: RETRO.panel,
      ...pixelBorderStyle(RETRO.neonGreen),
      fontFamily: "'Share Tech Mono', monospace",
      position: 'relative',
    }}>
      <PixelCorners />
      <div style={{ ...glowText(RETRO.neonGreen), fontSize: 12, marginBottom: 12, letterSpacing: '0.15em' }}>
        CLAUSECRAFT PIPELINE — EXECUTING
      </div>
      {PIPELINE_STEPS.slice(0, stepIdx).map((step, i) => (
        <div key={i} style={{
          color: i === stepIdx - 1 ? RETRO.neonGreen : RETRO.textMuted,
          fontSize: 11,
          marginBottom: 4,
          textShadow: i === stepIdx - 1 ? `0 0 6px ${RETRO.neonGreen}` : 'none',
          transition: 'all 0.2s',
        }}>
          {step}
          {i === stepIdx - 1 && <span style={{ animation: 'blink 0.8s infinite' }}> ▋</span>}
          {i < stepIdx - 1 && <span style={{ color: RETRO.neonGreen, marginLeft: 8 }}>✓</span>}
        </div>
      ))}
    </div>
  );
}

// ─── Markdown renderer (maps AI claims to retrieved chunks) ──────────────────
function ExplainableMarkdown({ text, passages, entities }) {
  if (!text) return null;

  // Simple: split by sentences and mark those matching a chunk
  const chunkTexts = (passages || []).map(p => p?.text?.toLowerCase() || '');

  const sentences = text.split(/(?<=[.!?])\s+/);
  return (
    <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 13, lineHeight: 2, color: RETRO.textPrimary }}>
      {sentences.map((sentence, i) => {
        const chunkIdx = chunkTexts.findIndex(ct => {
          const words = sentence.toLowerCase().split(/\W+/).filter(w => w.length > 4);
          return words.some(w => ct.includes(w));
        });
        const isGrounded = chunkIdx >= 0;
        return (
          <span
            key={i}
            title={isGrounded ? `Grounded in: Source [${chunkIdx + 1}]` : 'Unverified claim'}
            style={{
              display: 'inline',
              background: isGrounded ? '#0a200a' : 'transparent',
              borderBottom: isGrounded ? `1px dashed ${RETRO.neonGreen}88` : `1px dashed ${RETRO.neonMagenta}44`,
              cursor: isGrounded ? 'pointer' : 'default',
              padding: '1px 2px',
              marginRight: 4,
            }}
          >
            {sentence}{' '}
            {isGrounded && (
              <sup style={{ color: RETRO.neonCyan, fontSize: 9, fontWeight: 'bold' }}>
                [src:{chunkIdx + 1}]
              </sup>
            )}
          </span>
        );
      })}
    </div>
  );
}

// ─── Main QueryPage component ─────────────────────────────────────────────────
const QueryPage = () => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('answer');
  const flicker = useFlicker();

  const API_BASE = 'http://localhost:8000/api';

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setActiveTab('answer');

    try {
      const resp = await axios.post(`${API_BASE}/query`, {
        question: query,
        top_k: 5,
        use_entity_retrieval: true,
        use_reranking: true,
      });
      setResult(resp.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'SYSTEM ERROR — check backend connection.');
    } finally {
      setLoading(false);
    }
  };

  // Parse the nested JSON answer from rag_generator
  let extracted = null;
  let rawPassages = [];
  let rawEntities = [];

  if (result) {
    rawPassages = result.passages || result.retrieved_passages || [];
    rawEntities = result.highlighted_entities || result.entities || [];

    try {
      extracted = typeof result.answer === 'string'
        ? JSON.parse(result.answer)
        : result.answer;
    } catch {
      extracted = {
        simple_answer: result.answer || '',
        supporting_text: '',
        key_entities: [],
        confidence: result.confidence || 0,
        web_augmented: false,
        source: 'local',
      };
    }

    // Merge entity data from passages into top-level if not present
    if (!rawEntities.length && rawPassages.length) {
      rawEntities = rawPassages.flatMap(p => p.entities || []);
    }
  }

  const TABS = [
    { id: 'answer',   label: '▸ ANSWER',    color: RETRO.neonGreen   },
    { id: 'evidence', label: '▸ EVIDENCE',  color: RETRO.neonCyan    },
    { id: 'entities', label: '▸ ENTITIES',  color: RETRO.neonMagenta },
    { id: 'debug',    label: '▸ DEBUG',     color: RETRO.neonYellow  },
  ];

  return (
    <>
      {/* Google Fonts */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');
        @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0; } }
        @keyframes glitch {
          0%   { clip-path: inset(40% 0 61% 0); transform: translate(-4px, 0); }
          20%  { clip-path: inset(92% 0 1%  0); transform: translate( 4px, 0); }
          40%  { clip-path: inset(43% 0 1%  0); transform: translate(-2px, 0); }
          60%  { clip-path: inset(25% 0 58% 0); transform: translate( 2px, 0); }
          80%  { clip-path: inset(54% 0 7%  0); transform: translate(-1px, 0); }
          100% { clip-path: inset(58% 0 43% 0); transform: translate( 1px, 0); }
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: ${RETRO.bg}; }
        ::-webkit-scrollbar-thumb { background: ${RETRO.neonGreen}44; }
        ::-webkit-scrollbar-thumb:hover { background: ${RETRO.neonGreen}88; }
      `}</style>

      <ScanlineOverlay />

      <div style={{
        minHeight: '100vh',
        background: RETRO.bg,
        padding: '24px 16px',
        opacity: flicker,
      }}>

        {/* ── Header ─────────────────────────────────────────────────────── */}
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <div style={{
            fontFamily: "'Orbitron', monospace",
            fontSize: 32,
            fontWeight: 900,
            letterSpacing: '0.15em',
            ...glowText(RETRO.neonGreen),
            marginBottom: 4,
          }}>
            ▓▓ CLAUSECRAFT ▓▓
          </div>
          <div style={{
            fontFamily: "'Share Tech Mono', monospace",
            fontSize: 12,
            color: RETRO.textMuted,
            letterSpacing: '0.3em',
          }}>
            EXPLAINABLE LEGAL RAG SYSTEM · AGENTIC EDITION · v2.0
          </div>
        </div>

        {/* ── Search input ────────────────────────────────────────────────── */}
        <div style={{ maxWidth: 860, margin: '0 auto 28px' }}>
          <form onSubmit={handleSearch} style={{ position: 'relative' }}>
            <div style={{
              position: 'relative',
              ...pixelBorderStyle(RETRO.neonGreen),
              background: RETRO.panel,
            }}>
              <PixelCorners />
              <div style={{ display: 'flex', alignItems: 'center', padding: '4px 12px' }}>
                <span style={{ color: RETRO.neonGreen, fontFamily: 'monospace', marginRight: 8, fontSize: 16 }}>
                  ❯
                </span>
                <input
                  id="legal-query-input"
                  type="text"
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  placeholder="ENTER LEGAL QUERY_ (e.g. what is the punishment for theft under IPC section 378?)"
                  disabled={loading}
                  style={{
                    flex: 1,
                    background: 'transparent',
                    border: 'none',
                    outline: 'none',
                    color: RETRO.neonGreen,
                    fontFamily: "'Share Tech Mono', monospace",
                    fontSize: 14,
                    padding: '12px 0',
                    letterSpacing: '0.03em',
                    caretColor: RETRO.neonGreen,
                  }}
                />
                <button
                  id="query-submit-btn"
                  type="submit"
                  disabled={loading || !query.trim()}
                  style={{
                    background: loading ? RETRO.dimGreen : RETRO.neonGreen,
                    border: 'none',
                    color: RETRO.bg,
                    fontFamily: "'Orbitron', monospace",
                    fontWeight: 700,
                    fontSize: 11,
                    padding: '8px 20px',
                    cursor: loading ? 'wait' : 'pointer',
                    letterSpacing: '0.1em',
                    boxShadow: loading ? 'none' : `0 0 12px ${RETRO.neonGreen}`,
                    transition: 'all 0.2s',
                  }}
                >
                  {loading ? 'PROCESSING' : 'EXECUTE ▶'}
                </button>
              </div>
            </div>

            {/* Example queries */}
            <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
              {[
                'punishment for murder under IPC',
                'bail conditions under CrPC 438',
                'fundamental rights Article 21',
              ].map(q => (
                <button
                  key={q}
                  type="button"
                  onClick={() => setQuery(q)}
                  style={{
                    background: 'none',
                    border: `1px solid ${RETRO.textDim}`,
                    color: RETRO.textMuted,
                    fontFamily: "'Share Tech Mono', monospace",
                    fontSize: 10,
                    padding: '3px 10px',
                    cursor: 'pointer',
                    letterSpacing: '0.05em',
                    transition: 'all 0.15s',
                  }}
                  onMouseEnter={e => {
                    e.target.style.borderColor = RETRO.neonGreen;
                    e.target.style.color = RETRO.neonGreen;
                  }}
                  onMouseLeave={e => {
                    e.target.style.borderColor = RETRO.textDim;
                    e.target.style.color = RETRO.textMuted;
                  }}
                >
                  {q}
                </button>
              ))}
            </div>
          </form>
        </div>

        <div style={{ maxWidth: 860, margin: '0 auto' }}>

          {/* ── Error ─────────────────────────────────────────────────────── */}
          {error && (
            <div style={{
              padding: 16,
              ...pixelBorderStyle(RETRO.neonMagenta),
              background: RETRO.dimMagenta,
              color: RETRO.neonMagenta,
              fontFamily: "'Share Tech Mono', monospace",
              fontSize: 13,
              marginBottom: 20,
            }}>
              <PixelCorners color={RETRO.neonMagenta} />
              ⚠ SYSTEM FAULT: {error}
            </div>
          )}

          {/* ── Loading ────────────────────────────────────────────────────── */}
          {loading && <LoadingTerminal />}

          {/* ── Results ───────────────────────────────────────────────────── */}
          {extracted && !loading && (
            <div>
              {/* Confidence gauge + source tag */}
              <div style={{
                padding: 16,
                marginBottom: 16,
                background: RETRO.panel,
                ...pixelBorderStyle(RETRO.neonGreen, 1),
                position: 'relative',
              }}>
                <PixelCorners />
                <ConfidenceGauge
                  score={extracted.confidence || 0}
                  webAugmented={extracted.web_augmented || false}
                />
                {extracted.source && (
                  <div style={{
                    marginTop: 8,
                    fontFamily: 'monospace',
                    fontSize: 10,
                    color: RETRO.textMuted,
                    letterSpacing: '0.1em',
                  }}>
                    ROUTE: {(extracted.source || 'LOCAL').toUpperCase()} ·
                    PASSAGES RETRIEVED: {rawPassages.length}
                  </div>
                )}
              </div>

              {/* Tab bar */}
              <div style={{ display: 'flex', gap: 2, marginBottom: 2 }}>
                {TABS.map(tab => (
                  <button
                    key={tab.id}
                    id={`tab-${tab.id}`}
                    onClick={() => setActiveTab(tab.id)}
                    style={{
                      background: activeTab === tab.id ? tab.color : 'transparent',
                      border: `1px solid ${tab.color}`,
                      color: activeTab === tab.id ? RETRO.bg : tab.color,
                      fontFamily: "'Share Tech Mono', monospace",
                      fontSize: 11,
                      padding: '6px 16px',
                      cursor: 'pointer',
                      letterSpacing: '0.1em',
                      boxShadow: activeTab === tab.id ? `0 0 10px ${tab.color}` : 'none',
                      transition: 'all 0.15s',
                    }}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Tab panels */}
              <div style={{
                padding: 20,
                background: RETRO.panel,
                ...pixelBorderStyle(RETRO.panelBorder, 1),
                position: 'relative',
                minHeight: 280,
              }}>
                <PixelCorners color={RETRO.panelBorder} />

                {/* ANSWER TAB */}
                {activeTab === 'answer' && (
                  <div id="panel-answer">
                    <div style={{ ...glowText(RETRO.neonGreen), fontFamily: 'Orbitron', fontSize: 11, letterSpacing: '0.2em', marginBottom: 16 }}>
                      ▌ AI ANSWER (CLAIM-GROUNDED)
                    </div>
                    <ExplainableMarkdown
                      text={extracted.simple_answer}
                      passages={rawPassages}
                      entities={rawEntities}
                    />
                    {extracted.supporting_text && (
                      <div style={{ marginTop: 20 }}>
                        <div style={{ color: RETRO.textMuted, fontFamily: 'monospace', fontSize: 11, marginBottom: 8, letterSpacing: '0.1em' }}>
                          ▸ SUPPORTING CONTEXT
                        </div>
                        <div style={{
                          padding: 12,
                          background: '#070712',
                          border: `1px solid ${RETRO.textDim}`,
                          maxHeight: 160,
                          overflowY: 'auto',
                          color: RETRO.textMuted,
                          fontFamily: "'Share Tech Mono', monospace",
                          fontSize: 11,
                          lineHeight: 1.8,
                        }}>
                          {extracted.supporting_text}
                        </div>
                      </div>
                    )}
                    <JsonInspector data={result} />
                  </div>
                )}

                {/* EVIDENCE TAB */}
                {activeTab === 'evidence' && (
                  <div id="panel-evidence">
                    <div style={{ ...glowText(RETRO.neonCyan), fontFamily: 'Orbitron', fontSize: 11, letterSpacing: '0.2em', marginBottom: 16 }}>
                      ▌ RETRIEVED EVIDENCE CHUNKS ({rawPassages.length})
                    </div>
                    {rawPassages.length === 0 ? (
                      <p style={{ color: RETRO.textMuted, fontFamily: 'monospace', fontSize: 12 }}>
                        NO PASSAGES RETURNED — check backend retrieval pipeline.
                      </p>
                    ) : (
                      rawPassages.map((chunk, i) => (
                        <CitationCard
                          key={i}
                          chunk={chunk}
                          index={i}
                          isHighlighted={i === 0}
                          answer={extracted.simple_answer}
                        />
                      ))
                    )}
                  </div>
                )}

                {/* ENTITIES TAB */}
                {activeTab === 'entities' && (
                  <div id="panel-entities">
                    <div style={{ ...glowText(RETRO.neonMagenta), fontFamily: 'Orbitron', fontSize: 11, letterSpacing: '0.2em', marginBottom: 16 }}>
                      ▌ DETECTED LEGAL ENTITIES
                    </div>
                    {/* Grouped by label */}
                    {(() => {
                      const grouped = {};
                      const allEntities = [
                        ...(extracted.key_entities || []).map(e => {
                          const match = e.match(/^(.*?)\s*\(([^)]+)\)$/);
                          return match ? { text: match[1], label: match[2] } : { text: e, label: 'ENTITY' };
                        }),
                        ...rawEntities.map(e => typeof e === 'string'
                          ? { text: e, label: 'ENTITY' }
                          : e),
                      ];

                      allEntities.forEach(ent => {
                        const lbl = ent.label || 'ENTITY';
                        if (!grouped[lbl]) grouped[lbl] = [];
                        grouped[lbl].push(ent.text);
                      });

                      const entries = Object.entries(grouped);
                      if (!entries.length) {
                        return <p style={{ color: RETRO.textMuted, fontFamily: 'monospace', fontSize: 12 }}>NO ENTITIES DETECTED.</p>;
                      }

                      return entries.map(([label, texts]) => (
                        <div key={label} style={{ marginBottom: 16 }}>
                          <div style={{
                            fontFamily: 'monospace',
                            fontSize: 10,
                            color: (ENTITY_COLORS[label] || { fg: RETRO.textMuted }).fg,
                            letterSpacing: '0.15em',
                            marginBottom: 6,
                            borderBottom: `1px solid ${RETRO.panelBorder}`,
                            paddingBottom: 4,
                          }}>
                            ◈ {label}
                          </div>
                          <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                            {[...new Set(texts)].map((text, i) => (
                              <EntityBadge key={i} label={label} text={text} />
                            ))}
                          </div>
                        </div>
                      ));
                    })()}
                  </div>
                )}

                {/* DEBUG TAB */}
                {activeTab === 'debug' && (
                  <div id="panel-debug">
                    <div style={{ ...glowText(RETRO.neonYellow), fontFamily: 'Orbitron', fontSize: 11, letterSpacing: '0.2em', marginBottom: 16 }}>
                      ▌ RAW PIPELINE OUTPUT — TERMINAL VIEW
                    </div>
                    <div style={{
                      padding: 16,
                      background: '#020208',
                      ...pixelBorderStyle(RETRO.neonYellow, 1),
                      overflowX: 'auto',
                      maxHeight: 500,
                      overflowY: 'auto',
                    }}>
                      <pre style={{
                        fontFamily: "'Share Tech Mono', 'Courier New', monospace",
                        fontSize: 11,
                        color: RETRO.neonYellow,
                        whiteSpace: 'pre-wrap',
                        textShadow: `0 0 4px ${RETRO.neonYellow}88`,
                        lineHeight: 1.8,
                      }}>
                        {JSON.stringify(result, null, 2)}
                      </pre>
                    </div>
                    <div style={{ marginTop: 12, fontFamily: 'monospace', fontSize: 10, color: RETRO.textMuted }}>
                      {'// Tip: copy the above JSON to validate with the evaluation/metrics.py script.'}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── Empty state ────────────────────────────────────────────────── */}
          {!result && !loading && !error && (
            <div style={{
              textAlign: 'center',
              padding: 60,
              color: RETRO.textDim,
              fontFamily: "'Share Tech Mono', monospace",
              fontSize: 12,
              letterSpacing: '0.1em',
            }}>
              <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.3 }}>⚖</div>
              <div>AWAITING QUERY INPUT...</div>
              <div style={{ marginTop: 8, fontSize: 10 }}>
                DUAL-PIPELINE · ENTITY-AWARE · AGENTIC FALLBACK · ZERO HALLUCINATION
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default QueryPage;
