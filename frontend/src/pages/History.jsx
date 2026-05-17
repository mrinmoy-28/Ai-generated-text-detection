import React, { useState, useEffect } from 'react'
import { getHistory } from '../api/detector'
import toast from 'react-hot-toast'

export default function History() {
  const [history, setHistory] = useState([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const data = await getHistory()
        setHistory(data.history || [])
      } catch (error) {
        toast.error('Failed to load history')
      } finally {
        setIsLoading(false)
      }
    }

    fetchHistory()
  }, [])

  if (isLoading) {
    return (
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #ffffff 0%, #f9fafb 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: 'system-ui, -apple-system, sans-serif'
      }}>
        <p style={{ color: '#6b7280' }}>Loading history...</p>
      </div>
    )
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #ffffff 0%, #f9fafb 100%)',
      padding: '4rem 2rem',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{ marginBottom: '2rem' }}>
          <h1 style={{
            fontSize: '2.5rem',
            fontWeight: 800,
            color: '#1f2937',
            margin: '0 0 0.5rem 0'
          }}>
            History
          </h1>
          <p style={{
            fontSize: '1.05rem',
            color: '#6b7280',
            margin: 0
          }}>
            Your past detection analyses
          </p>
        </div>

        {/* Content */}
        {history.length === 0 ? (
          <div style={{
            background: '#ffffff',
            borderRadius: '12px',
            padding: '3rem 2rem',
            textAlign: 'center',
            border: '1px solid #e5e7eb',
            marginBottom: '2rem'
          }}>
            <p style={{
              fontSize: '1.1rem',
              color: '#6b7280',
              margin: 0
            }}>
              No submissions yet. Start by detecting some text!
            </p>
          </div>
        ) : (
          <div style={{
            background: '#ffffff',
            borderRadius: '12px',
            border: '1px solid #e5e7eb',
            overflow: 'hidden'
          }}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{
                width: '100%',
                borderCollapse: 'collapse'
              }}>
                <thead>
                  <tr style={{
                    borderBottom: '2px solid #e5e7eb',
                    backgroundColor: '#f9fafb'
                  }}>
                    <th style={{
                      padding: '1rem',
                      textAlign: 'left',
                      color: '#6b7280',
                      fontWeight: 600,
                      fontSize: '0.875rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em'
                    }}>
                      #
                    </th>
                    <th style={{
                      padding: '1rem',
                      textAlign: 'left',
                      color: '#6b7280',
                      fontWeight: 600,
                      fontSize: '0.875rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em'
                    }}>
                      Verdict
                    </th>
                    <th style={{
                      padding: '1rem',
                      textAlign: 'left',
                      color: '#6b7280',
                      fontWeight: 600,
                      fontSize: '0.875rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em'
                    }}>
                      Confidence
                    </th>
                    <th style={{
                      padding: '1rem',
                      textAlign: 'left',
                      color: '#6b7280',
                      fontWeight: 600,
                      fontSize: '0.875rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em'
                    }}>
                      Date
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {history.slice(0, 20).map((item, index) => (
                    <tr
                      key={item.id}
                      style={{
                        borderBottom: '1px solid #f0f0f0',
                        backgroundColor: index % 2 === 0 ? '#ffffff' : '#f9fafb',
                        transition: 'background-color 0.2s'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = '#f3f4f6'
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = index % 2 === 0 ? '#ffffff' : '#f9fafb'
                      }}
                    >
                      <td style={{
                        padding: '1rem',
                        color: '#6b7280',
                        fontWeight: 500
                      }}>
                        {index + 1}
                      </td>
                      <td style={{
                        padding: '1rem'
                      }}>
                        <span style={{
                          display: 'inline-block',
                          padding: '0.375rem 0.875rem',
                          borderRadius: '6px',
                          fontSize: '0.85rem',
                          fontWeight: 600,
                          backgroundColor: item.verdict === 'AI Generated' ? '#fee2e2' : '#dcfce7',
                          color: item.verdict === 'AI Generated' ? '#991b1b' : '#166534'
                        }}>
                          {item.verdict === 'AI Generated' ? '🤖 AI' : '✅ Human'}
                        </span>
                      </td>
                      <td style={{
                        padding: '1rem',
                        fontWeight: 600,
                        color: item.confidence >= 70
                          ? '#ef4444'
                          : item.confidence >= 40
                          ? '#f59e0b'
                          : '#10b981'
                      }}>
                        {item.confidence.toFixed(1)}%
                      </td>
                      <td style={{
                        padding: '1rem',
                        color: '#9ca3af',
                        fontSize: '0.9rem'
                      }}>
                        {new Date(item.timestamp).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {history.length > 20 && (
              <div style={{
                padding: '1rem',
                textAlign: 'center',
                borderTop: '1px solid #e5e7eb',
                backgroundColor: '#f9fafb',
                color: '#6b7280',
                fontSize: '0.9rem'
              }}>
                Showing 20 of {history.length} submissions
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
