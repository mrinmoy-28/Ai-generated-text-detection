import React from 'react'
import toast from 'react-hot-toast'

export default function ComparisonView({ text, result }) {
  // Safety checks
  if (!result || !text) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: '#6b7280' }}>
        <p>No data available for comparison</p>
      </div>
    )
  }

  const confidence = result.confidence || 0
  const verdict = result.verdict || 'Unknown'

  const copyToClipboard = (content) => {
    try {
      navigator.clipboard.writeText(content)
      toast.success('Copied to clipboard')
    } catch (error) {
      console.error('Copy failed:', error)
      toast.error('Failed to copy')
    }
  }

  return (
    <div style={{ width: '100%' }}>
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{
          fontSize: '1.5rem',
          fontWeight: 700,
          color: '#1f2937',
          margin: '0 0 0.5rem 0'
        }}>
          Analysis Breakdown
        </h2>
        <p style={{ color: '#6b7280', margin: 0, fontSize: '0.95rem' }}>
          Result: <span style={{ fontWeight: 600, color: '#1f2937' }}>{verdict}</span> ({confidence.toFixed(1)}% confidence)
        </p>
      </div>

      {/* Single Column - Clean Layout */}
      <div style={{
        background: '#ffffff',
        border: '1px solid #e5e7eb',
        borderRadius: '12px',
        padding: '2rem',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)'
      }}>
        <div style={{
          background: '#f3f4f6',
          padding: '1.5rem',
          borderRadius: '8px',
          marginBottom: '1.5rem',
          minHeight: '180px',
          maxHeight: '300px',
          overflowY: 'auto',
          lineHeight: '1.6',
          color: '#1f2937',
          whiteSpace: 'pre-wrap',
          wordWrap: 'break-word',
          fontSize: '0.95rem'
        }}>
          {text || 'No text provided'}
        </div>
        
        <button
          onClick={() => copyToClipboard(text)}
          style={{
            width: '100%',
            background: '#3b82f6',
            color: 'white',
            border: 'none',
            padding: '0.75rem 1rem',
            borderRadius: '8px',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.2s',
            fontSize: '0.95rem',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)'
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
          Copy Text
        </button>
      </div>
    </div>
  )
}
