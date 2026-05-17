import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { checkHealth } from '../api/detector'

export default function Navbar() {
  const [isOnline, setIsOnline] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        await checkHealth()
        setIsOnline(true)
      } catch (error) {
        setIsOnline(false)
      } finally {
        setLoading(false)
      }
    }

    checkBackendHealth()
    const interval = setInterval(checkBackendHealth, 10000)
    return () => clearInterval(interval)
  }, [])

  return (
    <nav style={{
      position: 'sticky',
      top: 0,
      zIndex: 50,
      background: '#ffffff',
      borderBottom: '1px solid #e5e7eb',
      boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 2rem' }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          height: '4rem'
        }}>
          {/* Logo/Brand */}
          <Link to="/" style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            fontWeight: 700,
            fontSize: '1.2rem',
            textDecoration: 'none',
            color: '#1f2937',
            transition: 'color 0.3s',
            letterSpacing: '-0.5px'
          }} 
            onMouseEnter={(e) => e.target.style.color = '#3b82f6'}
            onMouseLeave={(e) => e.target.style.color = '#1f2937'}
          >
            Detector
          </Link>

          {/* Navigation Links */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '3rem'
          }}>
            <Link to="/" style={{
              color: '#6b7280',
              textDecoration: 'none',
              fontSize: '0.9rem',
              fontWeight: 500,
              transition: 'color 0.3s',
              paddingBottom: '0.25rem'
            }} 
              onMouseEnter={(e) => e.target.style.color = '#3b82f6'}
              onMouseLeave={(e) => e.target.style.color = '#6b7280'}
            >
              Detect
            </Link>
            <Link to="/compare" style={{
              color: '#6b7280',
              textDecoration: 'none',
              fontSize: '0.9rem',
              fontWeight: 500,
              transition: 'color 0.3s',
              paddingBottom: '0.25rem'
            }}
              onMouseEnter={(e) => e.target.style.color = '#3b82f6'}
              onMouseLeave={(e) => e.target.style.color = '#6b7280'}
            >
              Compare
            </Link>
            <Link to="/batch" style={{
              color: '#6b7280',
              textDecoration: 'none',
              fontSize: '0.9rem',
              fontWeight: 500,
              transition: 'color 0.3s',
              paddingBottom: '0.25rem'
            }}
              onMouseEnter={(e) => e.target.style.color = '#3b82f6'}
              onMouseLeave={(e) => e.target.style.color = '#6b7280'}
            >
              Batch
            </Link>
            <Link to="/dashboard" style={{
              color: '#6b7280',
              textDecoration: 'none',
              fontSize: '0.9rem',
              fontWeight: 500,
              transition: 'color 0.3s',
              paddingBottom: '0.25rem'
            }}
              onMouseEnter={(e) => e.target.style.color = '#3b82f6'}
              onMouseLeave={(e) => e.target.style.color = '#6b7280'}
            >
              Analytics
            </Link>
            <Link to="/history" style={{
              color: '#6b7280',
              textDecoration: 'none',
              fontSize: '0.9rem',
              fontWeight: 500,
              transition: 'color 0.3s',
              paddingBottom: '0.25rem'
            }}
              onMouseEnter={(e) => e.target.style.color = '#3b82f6'}
              onMouseLeave={(e) => e.target.style.color = '#6b7280'}
            >
              History
            </Link>
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.4rem 1rem',
            backgroundColor: isOnline ? '#f0fdf4' : '#fef2f2',
            borderRadius: '8px',
            border: `1px solid ${isOnline ? '#dcfce7' : '#fecaca'}`,
            cursor: 'pointer',
            transition: 'all 0.2s'
          }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = isOnline ? '#dcfce7' : '#fee2e2'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = isOnline ? '#f0fdf4' : '#fef2f2'
            }}
          >
            <div style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              backgroundColor: isOnline ? '#16a34a' : '#dc2626',
              animation: !loading && isOnline ? 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite' : 'none'
            }}></div>
            <span style={{
              fontSize: '0.8rem',
              fontWeight: 500,
              color: isOnline ? '#15803d' : '#991b1b'
            }}>
              {loading ? '—' : isOnline ? 'Online' : 'Offline'}
            </span>
          </div>
        </div>
      </div>
    </nav>
  )
}
