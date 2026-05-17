import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Dashboard from './pages/Dashboard'
import History from './pages/History'
import Compare from './pages/Compare'
import Batch from './pages/Batch'

export default function App() {
  return (
    <Router>
      <div style={{ minHeight: '100vh', background: '#f9fafb', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/history" element={<History />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/batch" element={<Batch />} />
        </Routes>
        <Toaster
          position="top-right"
          toastOptions={{
            style: {
              background: '#ffffff',
              color: '#1f2937',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
            },
            success: {
              style: {
                background: '#dcfce7',
                color: '#166534',
                border: '1px solid #bbf7d0',
              },
            },
            error: {
              style: {
                background: '#fee2e2',
                color: '#991b1b',
                border: '1px solid #fca5a5',
              },
            },
          }}
        />
      </div>
    </Router>
  )
}
