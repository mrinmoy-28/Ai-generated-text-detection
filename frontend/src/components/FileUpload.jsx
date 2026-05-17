import React, { useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Loader2, Upload, File } from 'lucide-react'
import toast from 'react-hot-toast'

export default function FileUpload({ onSubmit, isLoading }) {
  const [selectedFile, setSelectedFile] = useState(null)

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0]
        const validTypes = ['text/plain', 'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
        const validExtensions = ['.txt', '.pdf', '.docx']

        const isValidType = validTypes.includes(file.type) || 
                           validExtensions.some(ext => file.name.toLowerCase().endsWith(ext))

        if (!isValidType) {
          toast.error('Only .txt, .pdf, and .docx files are allowed')
          return
        }

        setSelectedFile(file)
      }
    },
    accept: {
      'text/plain': ['.txt'],
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    multiple: false,
  })

  const handleAnalyze = () => {
    if (selectedFile) {
      onSubmit(selectedFile)
    }
  }

  const handleRemove = () => {
    setSelectedFile(null)
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
          <p className="text-primary font-semibold">Drop the file here...</p>
        ) : (
          <div>
            <p className="text-white font-semibold">Drag and drop your file here</p>
            <p className="text-slate-400 text-sm mt-1">or click to select from your computer</p>
            <p className="text-slate-500 text-xs mt-2">Supported: .txt, .pdf, .docx</p>
          </div>
        )}
      </div>

      {selectedFile && (
        <div className="bg-slate-800 p-4 rounded-lg flex items-center justify-between">
          <div className="flex items-center gap-3">
            <File className="w-5 h-5 text-primary" />
            <div>
              <p className="text-white font-medium">{selectedFile.name}</p>
              <p className="text-slate-400 text-sm">
                {(selectedFile.size / 1024).toFixed(2)} KB
              </p>
            </div>
          </div>
          <button
            onClick={handleRemove}
            disabled={isLoading}
            className="text-slate-400 hover:text-red-400 transition disabled:opacity-50"
          >
            ✕
          </button>
        </div>
      )}

      <button
        onClick={handleAnalyze}
        disabled={!selectedFile || isLoading}
        className={`w-full py-3 px-6 rounded-lg font-semibold transition flex items-center justify-center gap-2 ${
          selectedFile && !isLoading
            ? 'bg-primary hover:bg-blue-600 text-white cursor-pointer'
            : 'bg-slate-700 text-slate-400 cursor-not-allowed opacity-50'
        }`}
      >
        {isLoading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Analyzing...
          </>
        ) : (
          'Analyze File'
        )}
      </button>
    </div>
  )
}
