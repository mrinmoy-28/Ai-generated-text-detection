import React, { useState } from 'react'
import { Lightbulb, Loader2 } from 'lucide-react'
import { explainText } from '../api/detector'
import toast from 'react-hot-toast'

export default function ExplainabilityCard({ text }) {
  const [explanation, setExplanation] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isOpen, setIsOpen] = useState(false)

  const handleGetExplanation = async () => {
    setIsLoading(true)
    try {
      const result = await explainText(text)
      setExplanation(result)
      setIsOpen(true)
    } catch (error) {
      toast.error('Failed to get explanation')
    } finally {
      setIsLoading(false)
    }
  }

  if (isOpen && explanation) {
    return (
      <div className="animate-slideUp bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
          <Lightbulb className="w-6 h-6 text-yellow-500" />
          Why This Verdict?
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold text-red-400 mb-4">Words pushing toward AI</h4>
            <div className="space-y-2">
              {explanation.top_ai_words && explanation.top_ai_words.length > 0 ? (
                explanation.top_ai_words.map((item, index) => (
                  <div key={index} className="flex items-center justify-between bg-slate-700 p-3 rounded">
                    <span className="bg-red-500/20 text-red-300 px-3 py-1 rounded-full text-sm font-medium">
                      {item.word}
                    </span>
                    <span className="text-red-400 font-semibold">{item.score.toFixed(3)}</span>
                  </div>
                ))
              ) : (
                <p className="text-slate-400 text-sm">No AI-specific words detected</p>
              )}
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-green-400 mb-4">Words pushing toward Human</h4>
            <div className="space-y-2">
              {explanation.top_human_words && explanation.top_human_words.length > 0 ? (
                explanation.top_human_words.map((item, index) => (
                  <div key={index} className="flex items-center justify-between bg-slate-700 p-3 rounded">
                    <span className="bg-green-500/20 text-green-300 px-3 py-1 rounded-full text-sm font-medium">
                      {item.word}
                    </span>
                    <span className="text-green-400 font-semibold">{Math.abs(item.score).toFixed(3)}</span>
                  </div>
                ))
              ) : (
                <p className="text-slate-400 text-sm">No human-specific patterns detected</p>
              )}
            </div>
          </div>
        </div>

        <button
          onClick={() => setIsOpen(false)}
          className="mt-6 w-full py-2 px-4 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition"
        >
          Close Explanation
        </button>
      </div>
    )
  }

  return (
    <div className="animate-slideUp">
      <button
        onClick={handleGetExplanation}
        disabled={isLoading}
        className="w-full py-3 px-6 rounded-lg font-semibold transition flex items-center justify-center gap-2 bg-slate-700 hover:bg-slate-600 text-white"
      >
        {isLoading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Loading Explanation...
          </>
        ) : (
          <>
            <Lightbulb className="w-5 h-5" />
            Why This Verdict?
          </>
        )}
      </button>
    </div>
  )
}
