/*
  App.jsx — Voice Wine Explorer

  VOICE PIPELINE (browser side):
    1. User clicks mic → Web Speech API (SpeechRecognition) starts listening
    2. On result → transcript sent to POST /query (FastAPI backend)
    3. Backend returns { answer: string, sources: WineCard[] }
    4. Answer displayed as text AND spoken via Web Speech API (speechSynthesis)
    5. Wine cards rendered below the answer

  WHY WEB SPEECH API AND NOT WHISPER HERE?
    Web Speech API is free, zero-latency, and requires no server round-trip for STT.
    Whisper would be superior for accuracy but adds a second API call + latency.
    For a demo, Web Speech API is the right call — mention this tradeoff in your video.
    (Your Axon project used Whisper.cpp — you can speak to that experience directly.)

  BROWSER SUPPORT NOTE:
    SpeechRecognition works in Chrome/Edge. Firefox requires a flag.
    speechSynthesis works in all modern browsers.
    Show in Chrome for the demo.
*/

import { useState, useRef, useEffect, useCallback } from "react";
import "./App.css";

// ─── API ──────────────────────────────────────────────────────────────────────
const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

async function queryWines(question) {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: 5 }),
  });
  if (!res.ok) throw new Error(`Server error: ${res.status}`);
  return res.json(); // { answer, sources, query }
}

// ─── TTS ──────────────────────────────────────────────────────────────────────
function speak(text, onEnd) {
  window.speechSynthesis.cancel(); // stop any ongoing speech
  const utt = new SpeechSynthesisUtterance(text);
  utt.rate  = 0.95;
  utt.pitch = 1.0;
  // Pick a natural-sounding voice if available
  const voices = window.speechSynthesis.getVoices();
  const preferred = voices.find(
    (v) => v.name.includes("Samantha") || v.name.includes("Google US English") || v.lang === "en-US"
  );
  if (preferred) utt.voice = preferred;
  if (onEnd) utt.onend = onEnd;
  window.speechSynthesis.speak(utt);
}

// ─── SPEECH RECOGNITION HOOK ──────────────────────────────────────────────────
function useSpeechRecognition(onResult) {
  const recognitionRef = useRef(null);
  const [isListening, setIsListening] = useState(false);
  const [transcript,  setTranscript]  = useState("");

  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    const rec = new SpeechRecognition();
    rec.lang          = "en-US";
    rec.interimResults = true;   // show partial results as user speaks
    rec.maxAlternatives = 1;

    rec.onresult = (e) => {
      const current = Array.from(e.results)
        .map((r) => r[0].transcript)
        .join("");
      setTranscript(current);
      // Fire callback only on final result
      if (e.results[e.results.length - 1].isFinal) {
        onResult(current);
        setIsListening(false);
      }
    };

    rec.onerror = () => setIsListening(false);
    rec.onend   = () => setIsListening(false);

    recognitionRef.current = rec;
  }, [onResult]);

  const startListening = useCallback(() => {
    if (!recognitionRef.current) return;
    setTranscript("");
    setIsListening(true);
    recognitionRef.current.start();
  }, []);

  const stopListening = useCallback(() => {
    if (!recognitionRef.current) return;
    recognitionRef.current.stop();
    setIsListening(false);
  }, []);

  const supported = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
  return { isListening, transcript, startListening, stopListening, supported };
}

// ─── COMPONENTS ───────────────────────────────────────────────────────────────

function StarRating({ score }) {
  if (!score) return null;
  // Map 0-100 score to 0-5 stars
  const stars = Math.round((score / 100) * 5);
  return (
    <span className="stars" aria-label={`${score}/100`}>
      {"★".repeat(stars)}{"☆".repeat(5 - stars)}
      <span className="score-num">{score}</span>
    </span>
  );
}

function WineCard({ wine, index }) {
  const colorBadgeClass = {
    red:      "badge-red",
    white:    "badge-white",
    sparkling:"badge-sparkling",
  }[wine.color] || "badge-red";

  return (
    <article className="wine-card" style={{ "--delay": `${index * 0.07}s` }}>
      <div className="card-image-wrap">
        {wine.image_url ? (
          <img
            src={wine.image_url}
            alt={wine.name}
            className="card-image"
            loading="lazy"
            onError={(e) => { e.target.style.display = "none"; }}
          />
        ) : (
          <div className="card-image-placeholder">🍷</div>
        )}
        <span className={`color-badge ${colorBadgeClass}`}>{wine.color || "wine"}</span>
      </div>

      <div className="card-body">
        <p className="card-producer">{wine.producer}</p>
        <h3 className="card-name">{wine.name}</h3>

        <div className="card-meta">
          <span>{wine.region}{wine.country ? `, ${wine.country}` : ""}</span>
          {wine.vintage && <span>· {wine.vintage}</span>}
          {wine.varietal && <span>· {wine.varietal}</span>}
        </div>

        <div className="card-footer">
          <span className="card-price">${wine.price?.toFixed(2)}</span>
          {wine.avg_rating && <StarRating score={Math.round(wine.avg_rating)} />}
        </div>

        {wine.top_rating && (
          <blockquote className="card-note">
            "{wine.top_rating.note.slice(0, 100)}…"
            <cite>— {wine.top_rating.source}</cite>
          </blockquote>
        )}

        {wine.reference_url && (
          <a href={wine.reference_url} target="_blank" rel="noreferrer" className="card-link">
            View details →
          </a>
        )}
      </div>
    </article>
  );
}

function MicButton({ isListening, onClick, disabled }) {
  return (
    <button
      className={`mic-btn ${isListening ? "mic-active" : ""}`}
      onClick={onClick}
      disabled={disabled}
      aria-label={isListening ? "Listening…" : "Ask a question"}
    >
      <span className="mic-icon">{isListening ? "⏹" : "🎙"}</span>
      <span className="mic-label">{isListening ? "Listening…" : "Ask"}</span>
      {isListening && (
        <span className="mic-rings">
          <span /><span /><span />
        </span>
      )}
    </button>
  );
}

const EXAMPLE_QUESTIONS = [
  "Which are the best-rated wines under $50?",
  "What do you have from Burgundy?",
  "What's the most expensive bottle you have?",
  "Which bottles would make a good housewarming gift?",
  "I want something bold and Italian for dinner",
  "Recommend a Champagne for a celebration",
];

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function App() {
  const [question,   setQuestion]   = useState("");
  const [answer,     setAnswer]     = useState(null);
  const [sources,    setSources]    = useState([]);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const inputRef = useRef(null);

  const handleQuery = useCallback(async (q) => {
    const trimmed = q.trim();
    if (!trimmed) return;
    setQuestion(trimmed);
    setLoading(true);
    setError(null);
    setAnswer(null);
    setSources([]);

    try {
      const data = await queryWines(trimmed);
      setAnswer(data.answer);
      setSources(data.sources || []);
      // Speak the answer
      setIsSpeaking(true);
      speak(data.answer, () => setIsSpeaking(false));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const { isListening, transcript, startListening, stopListening, supported } =
    useSpeechRecognition(handleQuery);

  const handleMicClick = () => {
    if (isListening) stopListening();
    else startListening();
  };

  const handleTextSubmit = (e) => {
    e.preventDefault();
    const val = inputRef.current?.value?.trim();
    if (val) handleQuery(val);
  };

  const handleExampleClick = (q) => {
    if (inputRef.current) inputRef.current.value = q;
    handleQuery(q);
  };

  return (
    <div className="app">
      {/* ── HEADER ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">🍷</span>
            <div>
              <h1 className="logo-title">Sommelier</h1>
              <p className="logo-sub">Voice Wine Explorer</p>
            </div>
          </div>
        </div>
      </header>

      <main className="main">
        {/* ── SEARCH ZONE ── */}
        <section className="search-zone">
          <h2 className="search-heading">Ask me anything about our wine collection</h2>
          <p className="search-subheading">
            Speak your question or type it below — I'll find the perfect bottle.
          </p>

          <div className="input-row">
            <form onSubmit={handleTextSubmit} className="text-form">
              <input
                ref={inputRef}
                type="text"
                className="text-input"
                placeholder="e.g. Best Italian red under $30…"
                defaultValue=""
              />
              <button type="submit" className="submit-btn" disabled={loading}>
                {loading ? <span className="spinner" /> : "→"}
              </button>
            </form>

            {supported ? (
              <MicButton
                isListening={isListening}
                onClick={handleMicClick}
                disabled={loading}
              />
            ) : (
              <p className="no-speech">Voice not supported in this browser. Use Chrome.</p>
            )}
          </div>

          {/* Live transcript while speaking */}
          {isListening && transcript && (
            <p className="live-transcript">"{transcript}"</p>
          )}

          {/* Example chips */}
          <div className="example-chips">
            {EXAMPLE_QUESTIONS.map((q) => (
              <button
                key={q}
                className="chip"
                onClick={() => handleExampleClick(q)}
                disabled={loading || isListening}
              >
                {q}
              </button>
            ))}
          </div>
        </section>

        {/* ── ANSWER ZONE ── */}
        {(loading || answer || error) && (
          <section className="answer-zone">
            {loading && (
              <div className="loading-state">
                <div className="loading-dots">
                  <span /><span /><span />
                </div>
                <p>Consulting the cellar…</p>
              </div>
            )}

            {error && (
              <div className="error-state">
                <p>⚠ {error}</p>
                <p className="error-hint">Make sure the backend is running on port 8000.</p>
              </div>
            )}

            {answer && !loading && (
              <div className="answer-card">
                <div className="answer-header">
                  <span className="answer-icon">{isSpeaking ? "🔊" : "💬"}</span>
                  <span className="answer-label">
                    {isSpeaking ? "Speaking…" : `Re: "${question}"`}
                  </span>
                  <button
                    className="replay-btn"
                    onClick={() => { setIsSpeaking(true); speak(answer, () => setIsSpeaking(false)); }}
                    title="Replay answer"
                  >
                    ▶ Replay
                  </button>
                </div>
                <p className="answer-text">{answer}</p>
              </div>
            )}
          </section>
        )}

        {/* ── WINE CARDS ── */}
        {sources.length > 0 && !loading && (
          <section className="results-zone">
            <h3 className="results-heading">
              {sources.length} wine{sources.length !== 1 ? "s" : ""} from our collection
            </h3>
            <div className="cards-grid">
              {sources.map((wine, i) => (
                <WineCard key={wine.id} wine={wine} index={i} />
              ))}
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>Powered by Sentence Transformers · Gemini 1.5 Flash · Web Speech API</p>
      </footer>
    </div>
  );
}