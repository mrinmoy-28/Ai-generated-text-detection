import React, { useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Loader2, Upload, File, X } from 'lucide-react'
import toast from 'react-hot-toast'

export default function BatchUpload({ onSubmit, isLoading }) {
  const [files, setFiles] = useState([])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles) => {
      const validTypes = ['text/plain', 'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/zip']
      const validExtensions = ['.txt', '.pdf', '.docx', '.zip']

      const validFiles = acceptedFiles.filter((file) => {
        const isValidType = validTypes.includes(file.type) || 
                           validExtensions.some(ext => file.name.toLowerCase().endsWith(ext))
        if (!isValidType) {
          toast.error(`${file.name} is not a supported format`)
        }
        return isValidType
      })

      if (files.length + validFiles.length > 20) {
        toast.error('Maximum 20 files allowed')
        return
      }

      setFiles([...files, ...validFiles])
    },
    accept: {
      'text/plain': ['.txt'],
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/zip': ['.zip'],
    },
    multiple: true,
  })

  const handleAnalyze = () => {
    if (files.length > 0) {
      onSubmit(files)
    }
  }

  const handleRemoveFile = (index) => {
    setFiles(files.filter((_, i) => i !== index))
  }

  const handleClearAll = () => {
    setFiles([])
  }

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition ${
          isDragActive
            ? 'border-primary bg-primary/10'
            : 'border-slate-600 bg-slate-800/50 hover:bg-slate-800'
        }`}
      >
        <input {...getInputProps()} />
        <Upload className="w-12 h-12 mx-auto mb-3 text-slate-400" />
        {isDragActive ? (
          <p className="text-primary font-semibold">Drop the files here...</p>
        ) : (
          <div>
            <p className="text-white font-semibold">Drag and drop multiple files here</p>
            <p className="text-slate-400 text-sm mt-1">or click to select from your computer</p>
            <p className="text-slate-500 text-xs mt-2">Supported: .txt, .pdf, .docx, .zip (max 20 files)</p>
          </div>
        )}
      </div>

      {files.length > 0 && (
        <div className="bg-slate-800 p-4 rounded-lg">
          <div className="flex justify-between items-center mb-3">
            <p className="text-white font-semibold">
              {files.length} file{files.length !== 1 ? 's' : ''} selected
            </p>
            {files.length > 1 && (
              <button
                onClick={handleClearAll}
                disabled={isLoading}
                className="text-sm text-slate-400 hover:text-red-400 transition disabled:opacity-50"
              >
                Clear All
              </button>
            )}
          </div>

          <div className="space-y-2 max-h-48 overflow-y-auto">
            {files.map((file, index) => (
              <div key={index} className="flex items-center justify-between bg-slate-700 p-2 rounded">
                <div className="flex items-center gap-2">
                  <File className="w-4 h-4 text-primary" />
                  <div>
                    <p className="text-white text-sm font-medium">{file.name}</p>
                    <p className="text-slate-400 text-xs">
                      {(file.size / 1024).toFixed(2)} KB
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => handleRemoveFile(index)}
                  disabled={isLoading}
                  className="text-slate-400 hover:text-red-400 transition disabled:opacity-50"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      <button
        onClick={handleAnalyze}
        disabled={files.length === 0 || isLoading}
        className={`w-full py-3 px-6 rounded-lg font-semibold transition flex items-center justify-center gap-2 ${
          files.length > 0 && !isLoading
            ? 'bg-primary hover:bg-blue-600 text-white cursor-pointer'
            : 'bg-slate-700 text-slate-400 cursor-not-allowed opacity-50'
        }`}
      >
        {isLoading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Analyzing ({files.length})...
          </>
        ) : (
          `Analyze All (${files.length})`
        )}
      </button>
    </div>
  )
}
