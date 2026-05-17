import React, { useState } from 'react'
import toast from 'react-hot-toast'
import { compareDocuments, compareFiles, extractText } from '../api/detector'

export default function Compare() {
  const [mode, setMode] = useState('text') // 'text' or 'file'
  const [doc1Text, setDoc1Text] = useState('')
  const [doc2Text, setDoc2Text] = useState('')
  const [result, setResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleCompareText = async () => {
    if (!doc1Text.trim() || !doc2Text.trim()) {
      toast.error('Please enter text for both documents')
      return
    }

    if (doc1Text.trim().split(/\s+/).length < 20 || doc2Text.trim().split(/\s+/).length < 20) {
      toast.error('Each document must have at least 20 words')
      return
    }

    setIsLoading(true)
    try {
      const comparison = await compareDocuments(doc1Text, doc2Text)
      setResult(comparison)
      toast.success('Comparison complete!')
    } catch (error) {
      console.error('Comparison error:', error)
      toast.error(error?.response?.data?.detail || 'Comparison failed')
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileChange = async (e, docNum) => {
    const file = e.target.files?.[0]
    if (!file) return

    try {
      const extracted = await extractText(file)
      if (docNum === 1) {
        setDoc1Text(extracted.text || '')
      } else {
        setDoc2Text(extracted.text || '')
      }
      toast.success('File extracted successfully')
    } catch (error) {
      toast.error('Failed to extract text from file')
    }
  }

  const handleCompareFiles = async (e1, e2) => {
    const file1 = e1.target.files?.[0]
    const file2 = e2.target.files?.[0]

    if (!file1 || !file2) {
      toast.error('Please select both files')
      return
    }

    setIsLoading(true)
    try {
      console.log('Comparing files:', file1.name, file2.name)
      const comparison = await compareFiles(file1, file2)
      console.log('Comparison result:', comparison)
      setResult(comparison)
      toast.success('Comparison complete!')
    } catch (error) {
      console.error('File comparison error:', error)
      toast.error(error?.response?.data?.detail || 'File comparison failed')
    } finally {
      setIsLoading(false)
    }
  }

  if (result) {
    const similarity = result.similarity || 0
    const doc1Verdict = result.document1?.verdict
    const doc2Verdict = result.document2?.verdict
    const doc1Conf = result.document1?.confidence || 0
    const doc2Conf = result.document2?.confidence || 0

    return (
      <div style={{
        minHeight: '100vh',
        background: '#f9fafb',
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
              Comparison Results
            </h1>
            <p style={{
              fontSize: '1.05rem',
              color: '#6b7280',
              margin: 0
            }}>
              Side-by-side analysis of both documents
            </p>
          </div>

          {/* Similarity Score */}
          <div style={{
            background: '#ffffff',
            borderRadius: '12px',
            padding: '2rem',
            border: '1px solid #e5e7eb',
            marginBottom: '2rem',
            textAlign: 'center'
          }}>
            <h2 style={{
              fontSize: '1rem',
              color: '#6b7280',
              fontWeight: 600,
              margin: '0 0 1rem 0',
              textTransform: 'uppercase'
            }}>
              Content Similarity
            </h2>
            <p style={{
              fontSize: '3.5rem',
              fontWeight: 800,
              color: similarity > 70 ? '#ef4444' : similarity > 40 ? '#f59e0b' : '#10b981',
              margin: 0
            }}>
              {similarity.toFixed(1)}%
            </p>
          </div>

          {/* Documents Comparison */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '2rem',
            marginBottom: '2rem'
          }}>
            {/* Document 1 */}
            <div style={{
              background: '#ffffff',
              borderRadius: '12px',
              padding: '2rem',
              border: '1px solid #e5e7eb'
            }}>
              <h3 style={{
                fontSize: '1.05rem',
                fontWeight: 700,
                color: '#1f2937',
                margin: '0 0 1.5rem 0'
              }}>
                Document 1
              </h3>
              <div style={{
                textAlign: 'center',
                marginBottom: '1.5rem',
                padding: '1.5rem',
                backgroundColor: doc1Verdict === 'AI Generated' ? '#fee2e2' : '#dcfce7',
                borderRadius: '8px'
              }}>
                <p style={{
                  fontSize: '2rem',
                  margin: '0 0 0.5rem 0'
                }}>
                  {doc1Verdict === 'AI Generated' ? '🤖' : '✅'}
                </p>
                <p style={{
                  fontSize: '1.1rem',
                  fontWeight: 600,
                  color: doc1Verdict === 'AI Generated' ? '#991b1b' : '#166534',
                  margin: 0
                }}>
                  {doc1Verdict}
                </p>
                <p style={{
                  fontSize: '0.9rem',
                  color: doc1Verdict === 'AI Generated' ? '#b91c1c' : '#16a34a',
                  margin: '0.5rem 0 0 0'
                }}>
                  {doc1Conf.toFixed(1)}% confidence
                </p>
              </div>
            </div>

            {/* Document 2 */}
            <div style={{
              background: '#ffffff',
              borderRadius: '12px',
              padding: '2rem',
              border: '1px solid #e5e7eb'
            }}>
              <h3 style={{
                fontSize: '1.05rem',
                fontWeight: 700,
                color: '#1f2937',
                margin: '0 0 1.5rem 0'
              }}>
                Document 2
              </h3>
              <div style={{
                textAlign: 'center',
                marginBottom: '1.5rem',
                padding: '1.5rem',
                backgroundColor: doc2Verdict === 'AI Generated' ? '#fee2e2' : '#dcfce7',
                borderRadius: '8px'
              }}>
                <p style={{
                  fontSize: '2rem',
                  margin: '0 0 0.5rem 0'
                }}>
                  {doc2Verdict === 'AI Generated' ? '🤖' : '✅'}
                </p>
                <p style={{
                  fontSize: '1.1rem',
                  fontWeight: 600,
                  color: doc2Verdict === 'AI Generated' ? '#991b1b' : '#166534',
                  margin: 0
                }}>
                  {doc2Verdict}
                </p>
                <p style={{
                  fontSize: '0.9rem',
                  color: doc2Verdict === 'AI Generated' ? '#b91c1c' : '#16a34a',
                  margin: '0.5rem 0 0 0'
                }}>
                  {doc2Conf.toFixed(1)}% confidence
                </p>
              </div>
            </div>
          </div>

          {/* Action Button */}
          <button
            onClick={() => {
              setResult(null)
              setDoc1Text('')
              setDoc2Text('')
            }}
            style={{
              width: '100%',
              padding: '1rem',
              backgroundColor: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '1rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'background-color 0.2s'
            }}
            onMouseEnter={(e) => e.target.style.backgroundColor = '#2563eb'}
            onMouseLeave={(e) => e.target.style.backgroundColor = '#3b82f6'}
          >
            Compare Again
          </button>
        </div>
      </div>
    )
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: '#f9fafb',
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
            Compare Documents
          </h1>
          <p style={{
            fontSize: '1.05rem',
            color: '#6b7280',
            margin: 0
          }}>
            Analyze and compare two documents for AI-generated content
          </p>
        </div>

        {/* Mode Selector */}
        <div style={{
          display: 'flex',
          gap: '1rem',
          marginBottom: '2rem'
        }}>
          <button
            onClick={() => setMode('text')}
            style={{
              padding: '0.75rem 1.5rem',
              backgroundColor: mode === 'text' ? '#3b82f6' : '#f0f0f0',
              color: mode === 'text' ? 'white' : '#6b7280',
              border: 'none',
              borderRadius: '8px',
              fontWeight: 500,
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            📝 Text Mode
          </button>
          <button
            onClick={() => setMode('file')}
            style={{
              padding: '0.75rem 1.5rem',
              backgroundColor: mode === 'file' ? '#3b82f6' : '#f0f0f0',
              color: mode === 'file' ? 'white' : '#6b7280',
              border: 'none',
              borderRadius: '8px',
              fontWeight: 500,
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            📄 File Mode
          </button>
        </div>

        {/* Text Mode */}
        {mode === 'text' && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '2rem',
            marginBottom: '2rem'
          }}>
            {/* Document 1 */}
            <div style={{
              background: '#ffffff',
              borderRadius: '12px',
              padding: '2rem',
              border: '1px solid #e5e7eb'
            }}>
              <label style={{
                display: 'block',
                fontSize: '0.95rem',
                fontWeight: 600,
                color: '#1f2937',
                marginBottom: '0.75rem'
              }}>
                Document 1
              </label>
              <textarea
                value={doc1Text}
                onChange={(e) => setDoc1Text(e.target.value)}
                placeholder="Paste first document text here..."
                style={{
                  width: '100%',
                  minHeight: '200px',
                  padding: '1rem',
                  backgroundColor: '#f9fafb',
                  color: '#1f2937',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  fontSize: '0.9rem',
                  fontFamily: 'inherit',
                  outline: 'none',
                  boxSizing: 'border-box',
                  resize: 'vertical',
                  transition: 'all 0.2s'
                }}
                onFocus={(e) => {
                  e.target.style.backgroundColor = '#ffffff'
                  e.target.style.borderColor = '#3b82f6'
                }}
                onBlur={(e) => {
                  e.target.style.backgroundColor = '#f9fafb'
                  e.target.style.borderColor = '#e5e7eb'
                }}
              />
              <p style={{
                fontSize: '0.85rem',
                color: '#6b7280',
                margin: '0.5rem 0 0 0'
              }}>
                {doc1Text.trim().split(/\s+/).filter(w => w.length > 0).length} words
              </p>
            </div>

            {/* Document 2 */}
            <div style={{
              background: '#ffffff',
              borderRadius: '12px',
              padding: '2rem',
              border: '1px solid #e5e7eb'
            }}>
              <label style={{
                display: 'block',
                fontSize: '0.95rem',
                fontWeight: 600,
                color: '#1f2937',
                marginBottom: '0.75rem'
              }}>
                Document 2
              </label>
              <textarea
                value={doc2Text}
                onChange={(e) => setDoc2Text(e.target.value)}
                placeholder="Paste second document text here..."
                style={{
                  width: '100%',
                  minHeight: '200px',
                  padding: '1rem',
                  backgroundColor: '#f9fafb',
                  color: '#1f2937',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  fontSize: '0.9rem',
                  fontFamily: 'inherit',
                  outline: 'none',
                  boxSizing: 'border-box',
                  resize: 'vertical',
                  transition: 'all 0.2s'
                }}
                onFocus={(e) => {
                  e.target.style.backgroundColor = '#ffffff'
                  e.target.style.borderColor = '#3b82f6'
                }}
                onBlur={(e) => {
                  e.target.style.backgroundColor = '#f9fafb'
                  e.target.style.borderColor = '#e5e7eb'
                }}
              />
              <p style={{
                fontSize: '0.85rem',
                color: '#6b7280',
                margin: '0.5rem 0 0 0'
              }}>
                {doc2Text.trim().split(/\s+/).filter(w => w.length > 0).length} words
              </p>
            </div>
          </div>
        )}

        {/* File Mode */}
        {mode === 'file' && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '2rem',
            marginBottom: '2rem'
          }}>
            {/* File 1 */}
            <div style={{
              background: '#ffffff',
              borderRadius: '12px',
              padding: '2rem',
              border: '2px dashed #e5e7eb',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}>
              <input
                type="file"
                accept=".txt,.pdf,.docx"
                onChange={(e) => handleFileChange(e, 1)}
                style={{ display: 'none' }}
                id="file1"
              />
              <label htmlFor="file1" style={{
                display: 'block',
                cursor: 'pointer'
              }}>
                <p style={{ fontSize: '2rem', margin: '0 0 0.5rem 0' }}>📁</p>
                <p style={{
                  fontSize: '0.95rem',
                  fontWeight: 600,
                  color: '#1f2937',
                  margin: 0
                }}>
                  Upload Document 1
                </p>
                <p style={{
                  fontSize: '0.85rem',
                  color: '#6b7280',
                  margin: '0.25rem 0 0 0'
                }}>
                  TXT, PDF, DOCX
                </p>
              </label>
            </div>

            {/* File 2 */}
            <div style={{
              background: '#ffffff',
              borderRadius: '12px',
              padding: '2rem',
              border: '2px dashed #e5e7eb',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}>
              <input
                type="file"
                accept=".txt,.pdf,.docx"
                onChange={(e) => handleFileChange(e, 2)}
                style={{ display: 'none' }}
                id="file2"
              />
              <label htmlFor="file2" style={{
                display: 'block',
                cursor: 'pointer'
              }}>
                <p style={{ fontSize: '2rem', margin: '0 0 0.5rem 0' }}>📁</p>
                <p style={{
                  fontSize: '0.95rem',
                  fontWeight: 600,
                  color: '#1f2937',
                  margin: 0
                }}>
                  Upload Document 2
                </p>
                <p style={{
                  fontSize: '0.85rem',
                  color: '#6b7280',
                  margin: '0.25rem 0 0 0'
                }}>
                  TXT, PDF, DOCX
                </p>
              </label>
            </div>
          </div>
        )}

        {/* Compare Button */}
        <button
          onClick={mode === 'text' ? handleCompareText : handleCompareFiles}
          disabled={isLoading || (!doc1Text.trim() && !doc2Text.trim())}
          style={{
            width: '100%',
            padding: '1rem',
            backgroundColor: isLoading || (!doc1Text.trim() && !doc2Text.trim()) ? '#d1d5db' : '#3b82f6',
            color: isLoading || (!doc1Text.trim() && !doc2Text.trim()) ? '#9ca3af' : 'white',
            border: 'none',
            borderRadius: '8px',
            fontSize: '1rem',
            fontWeight: 600,
            cursor: isLoading || (!doc1Text.trim() && !doc2Text.trim()) ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s'
          }}
          onMouseEnter={(e) => {
            if (!isLoading && (doc1Text.trim() || doc2Text.trim())) {
              e.target.style.backgroundColor = '#2563eb'
            }
          }}
          onMouseLeave={(e) => {
            if (!isLoading && (doc1Text.trim() || doc2Text.trim())) {
              e.target.style.backgroundColor = '#3b82f6'
            }
          }}
        >
          {isLoading ? 'Comparing...' : 'Compare Documents'}
        </button>
      </div>
    </div>
  )
}
