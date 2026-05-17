import React, { useState, useEffect } from 'react'
import { getStats } from '../api/detector'
import toast from 'react-hot-toast'

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchStats()
  }, [])

  const fetchStats = async () => {
    try {
      const data = await getStats()
      setStats(data)
    } catch (error) {
      toast.error('Failed to load statistics')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        flexDirection: 'column',
        gap: '1rem'
      }}>
        <div style={{
          fontSize: '3rem'
        }}>⏳</div>
        <p style={{ color: '#475569', fontWeight: 600 }}>Loading statistics...</p>
      </div>
    )
  }

  if (!stats) {
    return (
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%)',
        padding: '4rem 2rem',
        fontFamily: 'system-ui, -apple-system, sans-serif'
      }}>
        <div style={{ maxWidth: '1000px', margin: '0 auto', textAlign: 'center' }}>
          <p style={{ color: '#64748b', fontSize: '1.1rem', fontWeight: 500 }}>No data available</p>
        </div>
      </div>
    )
  }

  const StatCard = ({ title, value, subtitle, color = '#3b82f6', icon }) => (
    <div style={{
      background: '#ffffff',
      borderRadius: '16px',
      padding: '2.5rem',
      border: '1px solid #e2e8f0',
      textAlign: 'center',
      boxShadow: '0 10px 30px rgba(0, 0, 0, 0.08)',
      transition: 'all 0.3s',
      position: 'relative',
      overflow: 'hidden',
      cursor: 'pointer'
    }}
      onMouseEnter={(e) => {
        e.currentTarget.style.boxShadow = '0 15px 40px rgba(0, 0, 0, 0.12)'
        e.currentTarget.style.transform = 'translateY(-4px)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.08)'
        e.currentTarget.style.transform = 'translateY(0)'
      }}
    >
      {/* Gradient overlay */}
      <div style={{
        position: 'absolute',
        top: 0,
        right: -40,
        width: '150px',
        height: '150px',
        background: `linear-gradient(135deg, ${color}, transparent)`,
        opacity: 0.05,
        borderRadius: '50%'
      }} />
      
      <div style={{
        fontSize: '3rem',
        marginBottom: '1.25rem',
        position: 'relative',
        zIndex: 1
      }}>
        {icon}
      </div>
      <h3 style={{
        color: '#64748b',
        fontSize: '0.85rem',
        fontWeight: 700,
        textTransform: 'uppercase',
        margin: '0 0 0.75rem 0',
        letterSpacing: '0.08em'
      }}>
        {title}
      </h3>
      <p style={{
        background: `linear-gradient(135deg, ${color}, ${color}dd)`,
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        backgroundClip: 'text',
        fontSize: '2.75rem',
        fontWeight: 900,
        margin: '1.25rem 0',
        position: 'relative',
        zIndex: 1,
        letterSpacing: '-1px'
      }}>
        {value}
      </p>
      {subtitle && (
        <p style={{
          color: '#94a3b8',
          fontSize: '0.9rem',
          margin: 0,
          fontWeight: 500,
          position: 'relative',
          zIndex: 1
        }}>
          {subtitle}
        </p>
      )}
    </div>
  )

  // Calculate totals from stats
  const total = stats.total_scans || 0
  const aiCount = stats.ai_detected || 0
  const humanCount = stats.human_detected || 0
  const aiPercent = total > 0 ? ((aiCount / total) * 100).toFixed(1) : 0

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%)',
      padding: '3.5rem 2rem',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{ marginBottom: '3.5rem' }}>
          <h1 style={{
            fontSize: '3rem',
            fontWeight: 900,
            background: 'linear-gradient(135deg, #3b82f6, #2563eb)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            margin: '0 0 0.5rem 0'
          }}>
            📊 Analytics Dashboard
          </h1>
          <p style={{
            fontSize: '1.1rem',
            color: '#64748b',
            margin: 0,
            fontWeight: 500
          }}>
            Your detection statistics and insights
          </p>
        </div>

        {/* Stats Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '2rem',
          marginBottom: '3.5rem'
        }}>
          <StatCard
            icon="📊"
            title="Total Scans"
            value={total}
            subtitle="All time detections"
            color="#3b82f6"
          />
          <StatCard
            icon="🤖"
            title="AI Generated"
            value={`${aiPercent}%`}
            subtitle={`${aiCount} out of ${total}`}
            color="#ef4444"
          />
          <StatCard
            icon="✅"
            title="Human Written"
            value={`${(100 - parseFloat(aiPercent)).toFixed(1)}%`}
            subtitle={`${humanCount} out of ${total}`}
            color="#10b981"
          />
          <StatCard
            icon="🎯"
            title="Avg Confidence"
            value={`${(stats.avg_confidence || 0).toFixed(1)}%`}
            subtitle="Detection accuracy"
            color="#f59e0b"
          />
        </div>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '1.5rem'
        }}>
          <div style={{
            background: '#eff6ff',
            borderRadius: '12px',
            padding: '1.5rem',
            border: '1px solid #bfdbfe'
          }}>
            <h3 style={{
              color: '#1e40af',
              fontSize: '1rem',
              fontWeight: 600,
              margin: '0 0 0.5rem 0'
            }}>
              💡 Pro Tip
            </h3>
            <p style={{
              color: '#1e40af',
              fontSize: '0.9rem',
              margin: 0
            }}>
              Analyze at least 20 words for the most accurate results
            </p>
          </div>

          <div style={{
            background: '#f0fdf4',
            borderRadius: '12px',
            padding: '1.5rem',
            border: '1px solid #bbf7d0'
          }}>
            <h3 style={{
              color: '#15803d',
              fontSize: '1rem',
              fontWeight: 600,
              margin: '0 0 0.5rem 0'
            }}>
              ✨ Features
            </h3>
            <p style={{
              color: '#15803d',
              fontSize: '0.9rem',
              margin: 0
            }}>
              Uses advanced ML models for high-accuracy detection
            </p>
          </div>

          <div style={{
            background: '#fef3c7',
            borderRadius: '12px',
            padding: '1.5rem',
            border: '1px solid #fde68a'
          }}>
            <h3 style={{
              color: '#92400e',
              fontSize: '1rem',
              fontWeight: 600,
              margin: '0 0 0.5rem 0'
            }}>
              🔒 Privacy
            </h3>
            <p style={{
              color: '#92400e',
              fontSize: '0.9rem',
              margin: 0
            }}>
              Your text is processed securely and never stored
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
