import React, { useState } from 'react'
import toast from 'react-hot-toast'
import { detectBatch } from '../api/detector'

export default function Batch() {
  const [files, setFiles] = useState([])
  const [results, setResults] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [progress, setProgress] = useState(0)

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files || [])
    if (selectedFiles.length === 0) return

    setFiles([...files, ...selectedFiles])
    toast.success(`${selectedFiles.length} file(s) added`)
  }

  const removeFile = (index) => {
    setFiles(files.filter((_, i) => i !== index))
  }

  const handleBatchUpload = async () => {
    if (files.length === 0) {
      toast.error('Please select at least one file')
      return
    }

    setIsLoading(true)
    setProgress(0)

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 10, 90))
      }, 200)

      const batchResults = await detectBatch(files)
      
      clearInterval(progressInterval)
      setProgress(100)
      setResults(batchResults)
      toast.success('Batch processing complete!')

      setTimeout(() => {
        setProgress(0)
      }, 1000)
    } catch (error) {
      toast.error('Batch processing failed')
      setProgress(0)
    } finally {
      setIsLoading(false)
    }
  }

  if (results) {
    const processedFiles = results.results || []
    const aiCount = processedFiles.filter(r => r.verdict === 'AI Generated').length
    const humanCount = processedFiles.length - aiCount

    return (
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #ffffff 0%, #f9fafb 100%)',
        padding: '4rem 2rem',
        fontFamily: 'system-ui, -apple-system, sans-serif'
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          {/* Header */}
          <div style={{ marginBottom: '2rem' }}>
            <h1 style={{
              fontSize: '2.5rem',
              fontWeight: 800,
              color: '#1f2937',
              margin: '0 0 0.5rem 0'
            }}>
              Batch Results
            </h1>
            <p style={{
              fontSize: '1.05rem',
              color: '#6b7280',
              margin: 0
            }}>
              Analysis of {processedFiles.length} documents
            </p>
          </div>

          {/* Summary Stats */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '1.5rem',
            marginBottom: '2rem'
          }}>
            <div style={{
              background: '#ffffff',
              borderRadius: '12px',
              padding: '1.5rem',
              border: '1px solid #e5e7eb',
              textAlign: 'center'
            }}>
              <p style={{ color: '#6b7280', fontSize: '0.875rem', fontWeight: 600, margin: 0, textTransform: 'uppercase' }}>
                Total Files
              </p>
              <p style={{ color: '#3b82f6', fontSize: '2.5rem', fontWeight: 800, margin: '0.5rem 0 0 0' }}>
                {processedFiles.length}
              </p>
            </div>
            <div style={{
              background: '#ffffff',
              borderRadius: '12px',
              padding: '1.5rem',
              border: '1px solid #e5e7eb',
              textAlign: 'center'
            }}>
              <p style={{ color: '#6b7280', fontSize: '0.875rem', fontWeight: 600, margin: 0, textTransform: 'uppercase' }}>
                AI Generated
              </p>
              <p style={{ color: '#ef4444', fontSize: '2.5rem', fontWeight: 800, margin: '0.5rem 0 0 0' }}>
                {aiCount}
              </p>
            </div>
            <div style={{
              background: '#ffffff',
              borderRadius: '12px',
              padding: '1.5rem',
              border: '1px solid #e5e7eb',
              textAlign: 'center'
            }}>
              <p style={{ color: '#6b7280', fontSize: '0.875rem', fontWeight: 600, margin: 0, textTransform: 'uppercase' }}>
                Human Written
              </p>
              <p style={{ color: '#10b981', fontSize: '2.5rem', fontWeight: 800, margin: '0.5rem 0 0 0' }}>
                {humanCount}
              </p>
            </div>
          </div>

          {/* Results Table */}
          <div style={{
            background: '#ffffff',
            borderRadius: '12px',
            border: '1px solid #e5e7eb',
            overflow: 'hidden',
            marginBottom: '2rem'
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
                      textTransform: 'uppercase'
                    }}>
                      File Name
                    </th>
                    <th style={{
                      padding: '1rem',
                      textAlign: 'left',
                      color: '#6b7280',
                      fontWeight: 600,
                      fontSize: '0.875rem',
                      textTransform: 'uppercase'
                    }}>
                      Verdict
                    </th>
                    <th style={{
                      padding: '1rem',
                      textAlign: 'left',
                      color: '#6b7280',
                      fontWeight: 600,
                      fontSize: '0.875rem',
                      textTransform: 'uppercase'
                    }}>
                      Confidence
                    </th>
                    <th style={{
                      padding: '1rem',
                      textAlign: 'left',
                      color: '#6b7280',
                      fontWeight: 600,
                      fontSize: '0.875rem',
                      textTransform: 'uppercase'
                    }}>
                      Words
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {processedFiles.map((result, index) => (
                    <tr
                      key={index}
                      style={{
                        borderBottom: '1px solid #f0f0f0',
                        backgroundColor: index % 2 === 0 ? '#ffffff' : '#f9fafb'
                      }}
                    >
                      <td style={{
                        padding: '1rem',
                        color: '#1f2937',
                        fontWeight: 500
                      }}>
                        {result.filename || `Document ${index + 1}`}
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
                          backgroundColor: result.verdict === 'AI Generated' ? '#fee2e2' : '#dcfce7',
                          color: result.verdict === 'AI Generated' ? '#991b1b' : '#166534'
                        }}>
                          {result.verdict === 'AI Generated' ? '🤖 AI' : '✅ Human'}
                        </span>
                      </td>
                      <td style={{
                        padding: '1rem',
                        fontWeight: 600,
                        color: result.confidence >= 70 ? '#ef4444' : result.confidence >= 40 ? '#f59e0b' : '#10b981'
                      }}>
                        {result.confidence.toFixed(1)}%
                      </td>
                      <td style={{
                        padding: '1rem',
                        color: '#6b7280',
                        fontSize: '0.9rem'
                      }}>
                        {result.word_count || 0}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Action Button */}
          <button
            onClick={() => {
              setResults(null)
              setFiles([])
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
            Process More Files
          </button>
        </div>
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
      <div style={{ maxWidth: '800px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{ marginBottom: '2rem' }}>
          <h1 style={{
            fontSize: '2.5rem',
            fontWeight: 800,
            color: '#1f2937',
            margin: '0 0 0.5rem 0'
          }}>
            Batch Upload
          </h1>
          <p style={{
            fontSize: '1.05rem',
            color: '#6b7280',
            margin: 0
          }}>
            Analyze multiple documents at once
          </p>
        </div>

        {/* Upload Area */}
        <div
          style={{
            background: '#ffffff',
            borderRadius: '12px',
            padding: '3rem 2rem',
            border: '2px dashed #e5e7eb',
            textAlign: 'center',
            marginBottom: '2rem',
            cursor: 'pointer',
            transition: 'all 0.2s'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = '#3b82f6'
            e.currentTarget.style.backgroundColor = '#eff6ff'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = '#e5e7eb'
            e.currentTarget.style.backgroundColor = '#ffffff'
          }}
        >
          <input
            type="file"
            multiple
            accept=".txt,.pdf,.docx"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
            id="batch-upload"
          />
          <label htmlFor="batch-upload" style={{
            display: 'block',
            cursor: 'pointer'
          }}>
            <p style={{ fontSize: '3rem', margin: '0 0 1rem 0' }}>📂</p>
            <p style={{
              fontSize: '1.1rem',
              fontWeight: 600,
              color: '#1f2937',
              margin: 0
            }}>
              Drop files here or click to select
            </p>
            <p style={{
              fontSize: '0.9rem',
              color: '#6b7280',
              margin: '0.5rem 0 0 0'
            }}>
              Supported: TXT, PDF, DOCX
            </p>
          </label>
        </div>

        {/* File List */}
        {files.length > 0 && (
          <div style={{
            background: '#ffffff',
            borderRadius: '12px',
            padding: '2rem',
            border: '1px solid #e5e7eb',
            marginBottom: '2rem'
          }}>
            <h3 style={{
              fontSize: '1rem',
              fontWeight: 600,
              color: '#1f2937',
              margin: '0 0 1rem 0'
            }}>
              Selected Files ({files.length})
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {files.map((file, index) => (
                <div
                  key={index}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '0.75rem',
                    backgroundColor: '#f9fafb',
                    borderRadius: '8px',
                    border: '1px solid #e5e7eb'
                  }}
                >
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem'
                  }}>
                    <span style={{ fontSize: '1.2rem' }}>📄</span>
                    <div>
                      <p style={{
                        fontSize: '0.9rem',
                        fontWeight: 500,
                        color: '#1f2937',
                        margin: 0
                      }}>
                        {file.name}
                      </p>
                      <p style={{
                        fontSize: '0.8rem',
                        color: '#9ca3af',
                        margin: '0.25rem 0 0 0'
                      }}>
                        {(file.size / 1024).toFixed(2)} KB
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => removeFile(index)}
                    style={{
                      padding: '0.5rem 0.75rem',
                      backgroundColor: '#fee2e2',
                      color: '#991b1b',
                      border: 'none',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '0.85rem',
                      fontWeight: 500,
                      transition: 'all 0.2s'
                    }}
                    onMouseEnter={(e) => e.target.style.backgroundColor = '#fca5a5'}
                    onMouseLeave={(e) => e.target.style.backgroundColor = '#fee2e2'}
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Progress Bar */}
        {isLoading && (
          <div style={{
            background: '#ffffff',
            borderRadius: '12px',
            padding: '2rem',
            border: '1px solid #e5e7eb',
            marginBottom: '2rem',
            textAlign: 'center'
          }}>
            <p style={{
              fontSize: '0.95rem',
              color: '#6b7280',
              margin: '0 0 1rem 0'
            }}>
              Processing {files.length} files...
            </p>
            <div style={{
              width: '100%',
              height: '8px',
              backgroundColor: '#e5e7eb',
              borderRadius: '4px',
              overflow: 'hidden',
              marginBottom: '1rem'
            }}>
              <div style={{
                height: '100%',
                width: `${progress}%`,
                backgroundColor: '#3b82f6',
                transition: 'width 0.3s ease-out'
              }} />
            </div>
            <p style={{
              fontSize: '0.85rem',
              color: '#9ca3af',
              margin: 0
            }}>
              {progress}%
            </p>
          </div>
        )}

        {/* Upload Button */}
        <button
          onClick={handleBatchUpload}
          disabled={isLoading || files.length === 0}
          style={{
            width: '100%',
            padding: '1rem',
            backgroundColor: isLoading || files.length === 0 ? '#d1d5db' : '#3b82f6',
            color: isLoading || files.length === 0 ? '#9ca3af' : 'white',
            border: 'none',
            borderRadius: '8px',
            fontSize: '1rem',
            fontWeight: 600,
            cursor: isLoading || files.length === 0 ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s'
          }}
          onMouseEnter={(e) => {
            if (!isLoading && files.length > 0) {
              e.target.style.backgroundColor = '#2563eb'
            }
          }}
          onMouseLeave={(e) => {
            if (!isLoading && files.length > 0) {
              e.target.style.backgroundColor = '#3b82f6'
            }
          }}
        >
          {isLoading ? 'Processing...' : `Process ${files.length} File${files.length !== 1 ? 's' : ''}`}
        </button>
      </div>
    </div>
  )
}
