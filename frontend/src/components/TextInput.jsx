import React, { useState } from 'react'
import { Loader2 } from 'lucide-react'

export default function TextInput({ onSubmit, isLoading }) {
  const [text, setText] = useState('')
  const [wordCount, setWordCount] = useState(0)

  const handleTextChange = (e) => {
    const newText = e.target.value
    setText(newText)
    const words = newText.trim().split(/\s+/).filter(word => word.length > 0)
    setWordCount(words.length)
  }

  const handleDetect = () => {
    if (text.trim().length === 0) {
      return
    }
    const words = text.trim().split(/\s+/).filter(w => w.length > 0)
    if (words.length < 20) {
      return
    }
    onSubmit(text)
  }

  const isTextValid = text.trim().length > 0 && wordCount >= 20

  return (
    <div className="space-y-4">
      <div className="relative">
        <textarea
          value={text}
          onChange={handleTextChange}
          placeholder="Paste your text here... (minimum 20 words)"
          className="w-full min-h-64 p-4 bg-slate-800 text-white border border-slate-600 rounded-lg focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/30 resize-none"
        />
      </div>

      <div className="flex justify-between items-center text-sm">
        <span className="text-slate-400">
          {wordCount} {wordCount === 1 ? 'word' : 'words'}
          {wordCount < 20 && wordCount > 0 && (
            <span className="text-yellow-400 ml-2">(minimum 20 required)</span>
          )}
        </span>
      </div>

      <button
        onClick={handleDetect}
        disabled={!isTextValid || isLoading}
        className={`w-full py-3 px-6 rounded-lg font-semibold transition flex items-center justify-center gap-2 ${
          isTextValid && !isLoading
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
          'Detect'
        )}
      </button>
    </div>
  )
}
