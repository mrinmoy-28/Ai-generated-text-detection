import React from 'react'

export default function ResultCard({ result }) {
  const isAI = result.verdict === 'AI Generated'
  const confidence = result.confidence

  const getColor = () => {
    if (confidence >= 70) return '#dc2626'
    if (confidence >= 40) return '#ea580c'
    return '#16a34a'
  }

  const getBg = () => {
    if (confidence >= 70) return '#fee2e2'
    if (confidence >= 40) return '#fed7aa'
    return '#dcfce7'
  }

  return (
    <div style={{
      background: '#ffffff',
      border: '1px solid #e5e7eb',
      borderRadius: '12px',
      padding: '2rem',
      textAlign: 'center'
    }}>
      <div style={{
        position: 'relative',
        width: '120px',
        height: '120px',
        margin: '0 auto 2rem'
      }}>
        <svg style={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          transform: 'rotate(-90deg)'
        }}>
          <circle cx="60" cy="60" r="50" stroke="#e5e7eb" strokeWidth="6" fill="none" />
          <circle
            cx="60"
            cy="60"
            r="50"
            stroke={getColor()}
            strokeWidth="6"
            fill="none"
            strokeDasharray={`${(confidence / 100) * 314} 314`}
            style={{ transition: 'stroke-dasharray 0.8s ease-out' }}
          />
        </svg>
        <div style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'column'
        }}>
          <span style={{
            fontSize: '2.5rem',
            fontWeight: 700,
            color: getColor()
          }}>
            {confidence.toFixed(0)}%
          </span>
        </div>
      </div>

      <h2 style={{
        fontSize: '1.75rem',
        fontWeight: 700,
        color: getColor(),
        margin: '0 0 0.5rem 0'
      }}>
        {isAI ? 'AI Generated' : 'Human Written'}
      </h2>

      <p style={{
        fontSize: '0.9rem',
        color: '#6b7280',
        margin: 0
      }}>
        Confidence Level
      </p>
    </div>
  )
}
