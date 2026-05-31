'use client'

import { useState, useRef, useEffect, useCallback } from 'react'

const API = 'http://localhost:8000'

// ── helpers ──────────────────────────────────────────────
function fmt(n) {
  if (n == null || n === 0 || n === '') return 'N/A'
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
  if (n >= 1_000)     return (n / 1_000).toFixed(1) + 'K'
  return Number(n).toLocaleString()
}

const CHIPS = [
  'Why did Video A get more engagement than Video B?',
  'What is the engagement rate of each?',
  'Compare the hooks in the first 5 seconds',
  'Who is the creator of Video B and their follower count?',
  'Suggest improvements for B based on what worked in A',
]

// ── sub-components ───────────────────────────────────────

function VideoCard({ label, platform, data }) {
  const isYT = platform === 'youtube'
  const color = isYT ? 'var(--yt)' : 'var(--ig)'
  const tag   = isYT ? 'YouTube' : 'Instagram'

  return (
    <div style={{
      background: 'var(--surface)',
      border: `1px solid var(--border)`,
      borderTop: `2px solid ${color}`,
      borderRadius: 6,
      padding: '18px 20px',
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      gap: 14,
      minHeight: 0,
    }}>
      {/* header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <span style={{
          background: color,
          color: '#080b0f',
          fontFamily: 'IBM Plex Mono',
          fontSize: 9,
          fontWeight: 700,
          letterSpacing: 2,
          padding: '3px 8px',
          borderRadius: 2,
        }}>
          VIDEO {label} · {tag}
        </span>
      </div>

      {/* title */}
      <p style={{
        fontFamily: 'Unbounded, sans-serif',
        fontSize: 12,
        fontWeight: 700,
        lineHeight: 1.5,
        color: 'var(--text)',
        display: '-webkit-box',
        WebkitLineClamp: 2,
        WebkitBoxOrient: 'vertical',
        overflow: 'hidden',
      }}>
        {data?.title || '—'}
      </p>

      {/* stats grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {[
          { label: 'VIEWS',    value: fmt(data?.views) },
          { label: 'LIKES',    value: fmt(data?.likes) },
          { label: isYT ? 'CHANNEL' : 'CREATOR', value: data?.channel || data?.creator || '—' },
          { label: 'ENG RATE', value: data ? (
            (data.views && data.views > 0)
              ? (((data.likes || 0)) / data.views * 100).toFixed(2) + '%'
              : 'N/A'
          ) : '—' },
        ].map(({ label: l, value: v }) => (
          <div key={l} style={{
            background: 'var(--surface2)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            padding: '8px 10px',
          }}>
            <div style={{ fontSize: 9, color: 'var(--muted2)', letterSpacing: 1.5, marginBottom: 4 }}>{l}</div>
            <div style={{
              fontSize: 12,
              fontWeight: 700,
              color: color,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}>{v}</div>
          </div>
        ))}
      </div>
    </div>
  )
}


function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: isUser ? 'flex-end' : 'flex-start',
      gap: 4,
      animation: 'msgIn 0.2s ease both',
    }}>
      <div style={{
        fontSize: 9,
        color: 'var(--muted)',
        letterSpacing: 1.5,
        textTransform: 'uppercase',
      }}>
        {isUser ? 'YOU' : 'RAG'}
      </div>
      <div style={{
        maxWidth: '82%',
        padding: '10px 14px',
        borderRadius: isUser ? '10px 10px 2px 10px' : '2px 10px 10px 10px',
        background: isUser ? 'var(--accent)' : 'var(--surface2)',
        border: isUser ? 'none' : '1px solid var(--border)',
        color: isUser ? '#080b0f' : 'var(--text)',
        fontSize: 13,
        lineHeight: 1.75,
        whiteSpace: 'pre-wrap',
        fontWeight: isUser ? 600 : 400,
      }}>
        {msg.text}
      </div>
    </div>
  )
}


function ThinkingBubble() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ fontSize: 9, color: 'var(--muted)', letterSpacing: 1.5 }}>RAG</div>
      <div style={{
        display: 'flex', gap: 5, alignItems: 'center',
        padding: '10px 14px',
        background: 'var(--surface2)',
        border: '1px solid var(--border)',
        borderRadius: '2px 10px 10px 10px',
        width: 'fit-content',
      }}>
        {[0, 1, 2].map(i => (
          <span key={i} style={{
            width: 6, height: 6,
            borderRadius: '50%',
            background: 'var(--accent)',
            display: 'inline-block',
            animation: `bounce 1.1s ${i * 0.15}s infinite ease-in-out`,
          }}/>
        ))}
      </div>
    </div>
  )
}

// ── main page ─────────────────────────────────────────────

export default function Home() {
  const [screen,      setScreen]      = useState('input')   // 'input' | 'dash'
  const [ytUrl,       setYtUrl]       = useState('')
  const [igUrl,       setIgUrl]       = useState('')
  const [videoA,      setVideoA]      = useState(null)
  const [videoB,      setVideoB]      = useState(null)
  const [extracting,  setExtracting]  = useState(false)
  const [extractErr,  setExtractErr]  = useState('')
  const [messages,    setMessages]    = useState([])
  const [chatInput,   setChatInput]   = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const messagesEndRef = useRef(null)
  const chatInputRef   = useRef(null)

  // auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, chatLoading])

  // ── extract ──────────────────────────────────────────────
  const handleExtract = useCallback(async () => {
    if (!ytUrl.trim() || !igUrl.trim()) {
      setExtractErr('Enter both URLs')
      return
    }
    setExtracting(true)
    setExtractErr('')

    try {
      const res = await fetch(`${API}/extract`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ youtube_url: ytUrl.trim(), instagram_url: igUrl.trim() }),
      })

      if (!res.ok) {
        const e = await res.json().catch(() => ({}))
        setExtractErr(e.detail || `Server error ${res.status}`)
        return
      }

      const data = await res.json()
      setVideoA(data.video_a)
      setVideoB(data.video_b)
      setMessages([])
      setScreen('dash')
    } catch (err) {
      setExtractErr('Cannot reach backend — is it running on port 8000?')
    } finally {
      setExtracting(false)
    }
  }, [ytUrl, igUrl])

  // ── chat ─────────────────────────────────────────────────
  const handleSend = useCallback(async (override) => {
    const q = (override ?? chatInput).trim()
    if (!q || chatLoading) return

    setChatInput('')
    setMessages(prev => [...prev, { role: 'user', text: q }])
    setChatLoading(true)

    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: q,
          video_a_id: videoA?.video_id ?? '',
          video_b_id: videoB?.video_id ?? '',
        }),
      })

      if (!res.ok) {
        const e = await res.json().catch(() => ({}))
        setMessages(prev => [...prev, { role: 'assistant', text: '⚠ ' + (e.detail || 'Server error') }])
        return
      }

      const data = await res.json()
      setMessages(prev => [...prev, { role: 'assistant', text: data.response }])
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', text: '⚠ Cannot reach backend.' }])
    } finally {
      setChatLoading(false)
      setTimeout(() => chatInputRef.current?.focus(), 50)
    }
  }, [chatInput, chatLoading, videoA, videoB])

  const handleKeyDown = useCallback(e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }, [handleSend])

  // ── INPUT SCREEN ─────────────────────────────────────────
  if (screen === 'input') return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 24,
    }}>
      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(16px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes msgIn {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
          40%            { transform: translateY(-6px); opacity: 1; }
        }
        @keyframes blink { 50% { opacity: 0; } }
      `}</style>

      <div style={{ width: '100%', maxWidth: 480, animation: 'fadeUp 0.4s ease both' }}>
        {/* wordmark */}
        <div style={{ marginBottom: 40 }}>
          <h1 style={{
            fontFamily: 'Unbounded, sans-serif',
            fontSize: 26,
            fontWeight: 900,
            letterSpacing: -1,
            color: 'var(--accent)',
          }}>
            RAG.COMPARE
          </h1>
          <p style={{ fontSize: 11, color: 'var(--muted2)', marginTop: 6, letterSpacing: 1 }}>
            // youtube + instagram · transcript rag · groq llm
          </p>
        </div>

        {/* URL inputs */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 14 }}>
          {[
            { id: 'yt',  tag: 'YT', color: 'var(--yt)', placeholder: 'https://youtube.com/watch?v=...', val: ytUrl, set: setYtUrl },
            { id: 'ig',  tag: 'IG', color: 'var(--ig)', placeholder: 'https://instagram.com/p/...', val: igUrl, set: setIgUrl },
          ].map(({ id, tag, color, placeholder, val, set }) => (
            <div key={id} style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <span style={{
                position: 'absolute', left: 12, zIndex: 1,
                background: color, color: '#080b0f',
                fontSize: 9, fontWeight: 700, letterSpacing: 1.5,
                padding: '3px 7px', borderRadius: 2,
                pointerEvents: 'none',
              }}>{tag}</span>
              <input
                value={val}
                onChange={e => set(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleExtract()}
                placeholder={placeholder}
                style={{
                  width: '100%',
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  borderRadius: 5,
                  color: 'var(--text)',
                  fontFamily: 'IBM Plex Mono, monospace',
                  fontSize: 12,
                  padding: '13px 14px 13px 64px',
                  outline: 'none',
                  transition: 'border-color 0.15s',
                }}
                onFocus={e => e.target.style.borderColor = color}
                onBlur={e => e.target.style.borderColor = 'var(--border)'}
              />
            </div>
          ))}
        </div>

        {/* extract button */}
        <button
          onClick={handleExtract}
          disabled={extracting}
          style={{
            width: '100%',
            padding: '14px',
            background: extracting ? 'var(--border)' : 'var(--accent)',
            color: extracting ? 'var(--muted)' : '#080b0f',
            border: 'none',
            borderRadius: 5,
            fontFamily: 'Unbounded, sans-serif',
            fontSize: 12,
            fontWeight: 700,
            letterSpacing: 1,
            cursor: extracting ? 'not-allowed' : 'pointer',
            transition: 'all 0.15s',
            textTransform: 'uppercase',
          }}
        >
          {extracting ? 'Extracting...' : 'Extract & Analyze →'}
        </button>

        {/* loading hint */}
        {extracting && (
          <p style={{ textAlign: 'center', fontSize: 11, color: 'var(--muted2)', marginTop: 12 }}>
            Whisper transcribing Instagram audio — ~30–60s
          </p>
        )}

        {/* error */}
        {extractErr && (
          <div style={{
            marginTop: 12, padding: '10px 14px',
            background: 'rgba(255,59,59,0.08)',
            border: '1px solid rgba(255,59,59,0.3)',
            borderRadius: 4, fontSize: 11, color: '#ff6b6b',
          }}>
            ⚠ {extractErr}
          </div>
        )}
      </div>
    </div>
  )

  // ── DASHBOARD ─────────────────────────────────────────────
  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(16px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes msgIn {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
          40%            { transform: translateY(-6px); opacity: 1; }
        }
      `}</style>

      {/* ── top bar ── */}
      <div style={{
        height: 48,
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px',
        flexShrink: 0,
      }}>
        <span style={{
          fontFamily: 'Unbounded, sans-serif',
          fontSize: 14,
          fontWeight: 900,
          color: 'var(--accent)',
          letterSpacing: -0.5,
        }}>RAG.COMPARE</span>

        <button
          onClick={() => { setScreen('input'); setVideoA(null); setVideoB(null); setMessages([]) }}
          style={{
            background: 'transparent',
            border: '1px solid var(--border)',
            color: 'var(--muted2)',
            fontFamily: 'IBM Plex Mono, monospace',
            fontSize: 10,
            padding: '5px 14px',
            borderRadius: 3,
            cursor: 'pointer',
            letterSpacing: 1,
            transition: 'all 0.15s',
          }}
          onMouseEnter={e => { e.target.style.borderColor = 'var(--accent)'; e.target.style.color = 'var(--accent)' }}
          onMouseLeave={e => { e.target.style.borderColor = 'var(--border)'; e.target.style.color = 'var(--muted2)' }}
        >
          ← NEW
        </button>
      </div>

      {/* ── main grid ── */}
      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: '300px 1fr',
        gap: 0,
        overflow: 'hidden',
        animation: 'fadeUp 0.35s ease both',
      }}>

        {/* ── left: video cards ── */}
        <div style={{
          borderRight: '1px solid var(--border)',
          display: 'flex',
          flexDirection: 'column',
          gap: 1,
          overflow: 'hidden',
        }}>
          {/* card A */}
          <div style={{ flex: 1, padding: 16, borderBottom: '1px solid var(--border)', overflow: 'auto' }}>
            <VideoCard label="A" platform="youtube" data={videoA} />
          </div>
          {/* card B */}
          <div style={{ flex: 1, padding: 16, overflow: 'auto' }}>
            <VideoCard label="B" platform="instagram" data={videoB} />
          </div>
        </div>

        {/* ── right: chat ── */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}>
          {/* chat header */}
          <div style={{
            padding: '12px 20px',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            flexShrink: 0,
          }}>
            <span style={{
              width: 7, height: 7, borderRadius: '50%',
              background: 'var(--accent)',
              boxShadow: '0 0 6px var(--accent)',
              animation: 'bounce 2s infinite',
            }}/>
            <span style={{ fontSize: 10, color: 'var(--muted2)', letterSpacing: 2, textTransform: 'uppercase' }}>
              RAG Chat — powered by Groq LLaMA 3.3 70B
            </span>
          </div>

          {/* messages */}
          <div style={{
            flex: 1,
            overflowY: 'auto',
            padding: '20px',
            display: 'flex',
            flexDirection: 'column',
            gap: 16,
          }}>

            {/* empty state with chips */}
            {messages.length === 0 && !chatLoading && (
              <div style={{
                margin: 'auto',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 16,
                textAlign: 'center',
              }}>
                <p style={{ fontSize: 11, color: 'var(--muted)', lineHeight: 2 }}>
                  Ask anything about these two videos
                </p>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', maxWidth: 520 }}>
                  {CHIPS.map(c => (
                    <button
                      key={c}
                      onClick={() => handleSend(c)}
                      style={{
                        background: 'transparent',
                        border: '1px solid var(--border)',
                        color: 'var(--muted2)',
                        fontFamily: 'IBM Plex Mono, monospace',
                        fontSize: 11,
                        padding: '6px 12px',
                        borderRadius: 20,
                        cursor: 'pointer',
                        transition: 'all 0.15s',
                        lineHeight: 1.4,
                      }}
                      onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--accent)'; e.currentTarget.style.background = 'rgba(0,255,135,0.05)' }}
                      onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--muted2)'; e.currentTarget.style.background = 'transparent' }}
                    >
                      {c}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) => <Message key={i} msg={m} />)}
            {chatLoading && <ThinkingBubble />}
            <div ref={messagesEndRef} />
          </div>

          {/* input bar */}
          <div style={{
            padding: '12px 16px',
            borderTop: '1px solid var(--border)',
            display: 'flex',
            gap: 10,
            flexShrink: 0,
            background: 'var(--surface)',
          }}>
            <input
              ref={chatInputRef}
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={chatLoading}
              placeholder="ask about engagement, hooks, improvements..."
              style={{
                flex: 1,
                background: 'var(--surface2)',
                border: '1px solid var(--border)',
                borderRadius: 4,
                color: 'var(--text)',
                fontFamily: 'IBM Plex Mono, monospace',
                fontSize: 12,
                padding: '10px 14px',
                outline: 'none',
                transition: 'border-color 0.15s',
              }}
              onFocus={e => e.target.style.borderColor = 'var(--accent)'}
              onBlur={e => e.target.style.borderColor = 'var(--border)'}
            />
            <button
              onClick={() => handleSend()}
              disabled={chatLoading || !chatInput.trim()}
              style={{
                padding: '10px 20px',
                background: (chatLoading || !chatInput.trim()) ? 'var(--border)' : 'var(--accent)',
                color: (chatLoading || !chatInput.trim()) ? 'var(--muted)' : '#080b0f',
                border: 'none',
                borderRadius: 4,
                fontFamily: 'Unbounded, sans-serif',
                fontSize: 11,
                fontWeight: 700,
                cursor: (chatLoading || !chatInput.trim()) ? 'not-allowed' : 'pointer',
                transition: 'all 0.15s',
                whiteSpace: 'nowrap',
              }}
            >
              SEND →
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
