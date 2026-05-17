import React, { useState } from 'react'
import toast from 'react-hot-toast'
import { detectText, detectFile, extractText } from '../api/detector'
import ComparisonView from '../components/ComparisonView'

export default function Home() {
  const [mode, setMode] = useState('text') // 'text' or 'file'
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleDetect = async () => {
    if (!text.trim()) {
      toast.error('Please enter some text')
      return
    }

    const words = text.trim().split(/\s+/).length
    if (words < 20) {
      toast.error('Minimum 20 words required')
      return
    }

    setIsLoading(true)
    try {
      const detectionResult = await detectText(text)
      setResult(detectionResult)
      toast.success('Detection complete')
    } catch (error) {
      toast.error('Backend not running at http://localhost:8000')
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    setIsLoading(true)
    try {
      // First extract text
      const extracted = await extractText(file)
      const fileText = extracted.text || ''

      // Then detect
      const detectionResult = await detectFile(file)
      setResult(detectionResult)
      setText(fileText)
      toast.success('File analysis complete')
    } catch (error) {
      toast.error('File processing failed')
    } finally {
      setIsLoading(false)
    }
  }

  if (result) {
    const isAI = result.verdict === 'AI Generated'
    const color = isAI ? '#ef4444' : '#10b981'
    const bgColor = isAI ? '#fef2f2' : '#f0fdf4'
    
    return (
      <div style={{
        minHeight: '100vh',
        background: '#f9fafb',
        padding: '3rem 2rem',
        fontFamily: 'system-ui, -apple-system, sans-serif'
      }}>
        <div style={{ maxWidth: '700px', margin: '0 auto' }}>
          {/* Result Card */}
          <div style={{
            background: '#ffffff',
            borderRadius: '12px',
            padding: '3rem 2.5rem',
            textAlign: 'center',
            border: `1px solid ${isAI ? '#fee2e2' : '#dcfce7'}`,
            marginBottom: '2rem',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
            position: 'relative',
            overflow: 'hidden'
          }}>
            {/* Gradient Background */}
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: '3px',
              background: isAI ? '#ef4444' : '#10b981',
              opacity: 1
            }} />
            
            <div style={{
              fontSize: '3rem',
              marginBottom: '1.25rem',
              opacity: 0.8
            }}>
              {isAI ? '⚠' : '✓'}
            </div>
            
            <h2 style={{
              fontSize: '2rem',
              fontWeight: 700,
              color: isAI ? '#dc2626' : '#16a34a',
              margin: '0 0 0.5rem 0'
            }}>
              {isAI ? 'AI Generated' : 'Human Written'}
            </h2>
            
            <p style={{
              fontSize: '1rem',
              color: '#64748b',
              margin: '0 0 2rem 0',
              fontWeight: 500
            }}>
              Confidence Level
            </p>

            {/* Confidence Circle */}
            <div style={{
              position: 'relative',
              width: '180px',
              height: '180px',
              margin: '0 auto 2rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <svg style={{
                position: 'absolute',
                width: '100%',
                height: '100%',
                opacity: 0.2
              }}>
                <circle cx="90" cy="90" r="80" stroke={color} strokeWidth="3" fill="none" />
              </svg>
              <div style={{
                fontSize: '3.5rem',
                fontWeight: 900,
                color: color,
                letterSpacing: '-2px'
              }}>
                {result.confidence.toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Comparison View */}
          <div style={{
            background: '#ffffff',
            borderRadius: '12px',
            padding: '2.5rem',
            marginBottom: '2rem',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
            border: '1px solid #e5e7eb'
          }}>
            <ComparisonView text={text} result={result} />
          </div>

          {/* Model Scores */}
          <div style={{
            background: '#ffffff',
            borderRadius: '12px',
            padding: '2.5rem',
            marginBottom: '2rem',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
            border: '1px solid #e5e7eb'
          }}>
            <h3 style={{
              fontSize: '1.1rem',
              fontWeight: 700,
              color: '#1f2937',
              marginTop: 0,
              marginBottom: '2rem',
              letterSpacing: '0.3px'
            }}>
              Model Analysis
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
              {Object.entries(result.breakdown).map(([model, score]) => (
                <div key={model}>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '0.75rem'
                  }}>
                    <span style={{
                      fontSize: '0.95rem',
                      fontWeight: 600,
                      color: '#1e293b',
                      textTransform: 'capitalize',
                      letterSpacing: '0.2px'
                    }}>
                      {model.replace(/_/g, ' ')}
                    </span>
                    <span style={{
                      fontSize: '1rem',
                      fontWeight: 700,
                      color: score >= 70 ? '#ef4444' : score >= 40 ? '#f59e0b' : '#10b981',
                      background: score >= 70 ? '#fef2f2' : score >= 40 ? '#fefce8' : '#f0fdf4',
                      padding: '0.4rem 0.9rem',
                      borderRadius: '8px'
                    }}>
                      {score.toFixed(1)}%
                    </span>
                  </div>
                  <div style={{
                    height: '8px',
                    backgroundColor: '#e2e8f0',
                    borderRadius: '4px',
                    overflow: 'hidden',
                    boxShadow: 'inset 0 2px 4px rgba(0, 0, 0, 0.05)'
                  }}>
                    <div style={{
                      height: '100%',
                      width: `${Math.min(score, 100)}%`,
                      backgroundColor: score >= 70 ? '#ef4444' : score >= 40 ? '#f59e0b' : '#10b981',
                      transition: 'width 0.8s ease-out',
                      borderRadius: '4px',
                      boxShadow: `0 0 8px ${score >= 70 ? 'rgba(239, 68, 68, 0.4)' : score >= 40 ? 'rgba(245, 158, 11, 0.4)' : 'rgba(16, 185, 129, 0.4)'}`
                    }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Actions */}
          <button
            onClick={() => {
              setResult(null)
              setText('')
            }}
            style={{
              width: '100%',
              padding: '1rem',
              background: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '1rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
              letterSpacing: '0.3px'
            }}
            onMouseEnter={(e) => {
              e.target.style.background = '#2563eb'
              e.target.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.15)'
            }}
            onMouseLeave={(e) => {
              e.target.style.background = '#3b82f6'
              e.target.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.1)'
            }}
          >
            Analyze Again
          </button>
        </div>
      </div>
    )
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: '#f9fafb',
      padding: '3rem 2rem',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      <div style={{ maxWidth: '700px', margin: '0 auto' }}>
        {/* Hero Section */}
        <div style={{ marginBottom: '2.5rem', textAlign: 'center' }}>
          <h1 style={{
            fontSize: '3rem',
            fontWeight: 700,
            color: '#1f2937',
            margin: '0 0 0.75rem 0'
          }}>
            AI Text Detector
          </h1>
          <p style={{
            fontSize: '1.1rem',
            color: '#6b7280',
            margin: 0,
            fontWeight: 400,
            letterSpacing: '0.3px'
          }}>
            Identify AI-generated text with precision
          </p>
        </div>

        {/* Mode Selector */}
        <div style={{
          display: 'flex',
          gap: '1rem',
          marginBottom: '2rem',
          justifyContent: 'center'
        }}>
          <button
            onClick={() => {
              setMode('text')
              setText('')
            }}
            style={{
              padding: '0.8rem 1.8rem',
              backgroundColor: mode === 'text' ? '#3b82f6' : '#ffffff',
              color: mode === 'text' ? 'white' : '#6b7280',
              border: '1px solid ' + (mode === 'text' ? '#3b82f6' : '#d1d5db'),
              borderRadius: '8px',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s',
              boxShadow: mode === 'text' ? '0 1px 3px rgba(0, 0, 0, 0.1)' : 'none',
              fontSize: '0.9rem'
            }}
            onMouseEnter={(e) => {
              if (mode !== 'text') {
                e.target.style.borderColor = '#3b82f6'
                e.target.style.backgroundColor = '#f0f9ff'
              }
            }}
            onMouseLeave={(e) => {
              if (mode !== 'text') {
                e.target.style.borderColor = '#d1d5db'
                e.target.style.backgroundColor = '#ffffff'
              }
            }}
          >
            Text Input
          </button>
          <button
            onClick={() => {
              setMode('file')
              setText('')
            }}
            style={{
              padding: '0.8rem 1.8rem',
              backgroundColor: mode === 'file' ? '#3b82f6' : '#ffffff',
              color: mode === 'file' ? 'white' : '#6b7280',
              border: '1px solid ' + (mode === 'file' ? '#3b82f6' : '#d1d5db'),
              borderRadius: '8px',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s',
              boxShadow: mode === 'file' ? '0 1px 3px rgba(0, 0, 0, 0.1)' : 'none',
              fontSize: '0.9rem'
            }}
            onMouseEnter={(e) => {
              if (mode !== 'file') {
                e.target.style.borderColor = '#3b82f6'
                e.target.style.backgroundColor = '#f0f9ff'
              }
            }}
            onMouseLeave={(e) => {
              if (mode !== 'file') {
                e.target.style.borderColor = '#d1d5db'
                e.target.style.backgroundColor = '#ffffff'
              }
            }}
          >
            File Upload
          </button>
        </div>

        {/* Input Box */}
        <div style={{
          background: '#ffffff',
          borderRadius: '12px',
          padding: '2.5rem',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          border: '1px solid #e5e7eb',
          transition: 'all 0.2s'
        }}
          onMouseEnter={(e) => {
            e.currentTarget.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.15)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.1)'
          }}
        >
          {mode === 'text' ? (
            <>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste your text here... (minimum 20 words)"
                style={{
                  width: '100%',
                  minHeight: '240px',
                  padding: '1.25rem',
                  backgroundColor: '#f8fafc',
                  color: '#0f172a',
                  border: '1.5px solid #e2e8f0',
                  borderRadius: '10px',
                  fontSize: '0.95rem',
                  fontFamily: 'system-ui, -apple-system, sans-serif',
                  outline: 'none',
                  boxSizing: 'border-box',
                  resize: 'vertical',
                  transition: 'all 0.3s',
                  fontWeight: 500
                }}
                onFocus={(e) => {
                  e.target.style.backgroundColor = '#ffffff'
                  e.target.style.borderColor = '#3b82f6'
                  e.target.style.boxShadow = '0 0 0 3px rgba(59, 130, 246, 0.15)'
                }}
                onBlur={(e) => {
                  e.target.style.backgroundColor = '#f8fafc'
                  e.target.style.borderColor = '#e2e8f0'
                  e.target.style.boxShadow = 'none'
                }}
              />

              {/* Word Count */}
              <div style={{
                marginTop: '1.25rem',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '0.75rem 1rem',
                backgroundColor: '#f0f4f8',
                borderRadius: '8px'
              }}>
                <span style={{
                  fontSize: '0.9rem',
                  color: '#475569',
                  fontWeight: 500
                }}>
                  {text.trim().split(/\s+/).filter(w => w.length > 0).length} words
                </span>
                {text.trim().split(/\s+/).filter(w => w.length > 0).length < 20 && text.trim() && (
                  <span style={{
                    fontSize: '0.9rem',
                    color: '#f59e0b',
                    fontWeight: 600
                  }}>
                    Need {20 - text.trim().split(/\s+/).filter(w => w.length > 0).length} more
                  </span>
                )}
              </div>

              {/* Analyze Button */}
              <button
                onClick={handleDetect}
                disabled={isLoading || !text.trim()}
                style={{
                  width: '100%',
                  marginTop: '1.75rem',
                  padding: '1rem',
                  background: isLoading || !text.trim() ? '#d1d5db' : '#3b82f6',
                  color: isLoading || !text.trim() ? '#9ca3af' : 'white',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '1rem',
                  fontWeight: 600,
                  cursor: isLoading || !text.trim() ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s',
                  boxShadow: isLoading || !text.trim() ? 'none' : '0 1px 3px rgba(0, 0, 0, 0.1)',
                  letterSpacing: '0.3px'
                }}
                onMouseEnter={(e) => {
                  if (!isLoading && text.trim()) {
                    e.target.style.background = '#2563eb'
                    e.target.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.15)'
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isLoading && text.trim()) {
                    e.target.style.background = '#3b82f6'
                    e.target.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.1)'
                  }
                }}
              >
                {isLoading ? 'Analyzing...' : 'Analyze'}
              </button>
            </>
          ) : (
            <>
              <div
                style={{
                  border: '2.5px dashed #cbd5e1',
                  borderRadius: '12px',
                  padding: '3rem',
                  textAlign: 'center',
                  minHeight: '240px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: isLoading ? 'not-allowed' : 'pointer',
                  transition: 'all 0.3s',
                  backgroundColor: '#f8fafc'
                }}
                onMouseEnter={(e) => {
                  if (!isLoading) {
                    e.currentTarget.style.borderColor = '#3b82f6'
                    e.currentTarget.style.backgroundColor = '#eff6ff'
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isLoading) {
                    e.currentTarget.style.borderColor = '#cbd5e1'
                    e.currentTarget.style.backgroundColor = '#f8fafc'
                  }
                }}
              >
                <input
                  type="file"
                  accept=".txt,.pdf,.docx"
                  onChange={handleFileUpload}
                  disabled={isLoading}
                  style={{ display: 'none' }}
                  id="file-upload"
                />
                <label htmlFor="file-upload" style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  cursor: 'pointer',
                  width: '100%'
                }}>
                  <p style={{ fontSize: '3rem', margin: 0 }}>📁</p>
                  <p style={{
                    fontSize: '1rem',
                    fontWeight: 600,
                    color: '#1f2937',
                    margin: '1rem 0 0.5rem 0'
                  }}>
                    Upload Document
                  </p>
                  <p style={{
                    fontSize: '0.9rem',
                    color: '#6b7280',
                    margin: 0
                  }}>
                    TXT, PDF, or DOCX
                  </p>
                </label>
              </div>

              <button
                style={{
                  width: '100%',
                  marginTop: '1.5rem',
                  padding: '1rem',
                  backgroundColor: isLoading ? '#d1d5db' : '#3b82f6',
                  color: isLoading ? '#9ca3af' : 'white',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '1rem',
                  fontWeight: 600,
                  cursor: isLoading ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s'
                }}
                disabled={isLoading || !text.trim()}
                onMouseEnter={(e) => {
                  if (!isLoading && text.trim()) {
                    e.target.style.backgroundColor = '#2563eb'
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isLoading && text.trim()) {
                    e.target.style.backgroundColor = '#3b82f6'
                  }
                }}
              >
                {isLoading ? 'Processing...' : 'Analyze File'}
              </button>
            </>
          )}
        </div>

        {/* Info */}
        <div style={{
          marginTop: '2rem',
          padding: '1.25rem',
          backgroundColor: '#eff6ff',
          borderRadius: '8px',
          border: '1px solid #bfdbfe'
        }}>
          <p style={{
            fontSize: '0.875rem',
            color: '#1e40af',
            margin: 0
          }}>
            💡 Uses multiple AI detection models for high accuracy
          </p>
        </div>
      </div>
    </div>
  )
}

