import React, { useState } from 'react'

export default function SentenceHighlight({ sentences, text }) {
  const [hoveredIndex, setHoveredIndex] = useState(null)

  if (!sentences || sentences.length === 0) {
    return null
  }

  return (
    <div className="animate-slideUp bg-slate-800 rounded-xl p-6 border border-slate-700">
      <h3 className="text-xl font-bold text-white mb-4">Sentence Analysis</h3>

      <div className="mb-4 flex gap-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-500 rounded"></div>
          <span className="text-slate-300">AI Generated</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-500 rounded"></div>
          <span className="text-slate-300">Human Written</span>
        </div>
      </div>

      <div className="bg-slate-700/50 p-4 rounded-lg leading-relaxed text-lg">
        {sentences.map((item, index) => {
          const aiScore = item.ai_score
          const bgOpacity = Math.min(aiScore / 100, 1)

          let bgColor = item.is_ai
            ? `rgba(239, 68, 68, ${bgOpacity * 0.3})`
            : `rgba(34, 197, 94, ${0.2})`

          return (
            <span
              key={index}
              className="relative group cursor-help transition"
              style={{ backgroundColor: bgColor }}
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex(null)}
            >
              {item.sentence}{' '}
              {hoveredIndex === index && (
                <span className="absolute bottom-full left-0 mb-2 px-3 py-1 bg-slate-900 text-slate-100 text-sm rounded whitespace-nowrap border border-slate-600 z-10">
                  AI Score: {aiScore.toFixed(1)}%
                </span>
              )}
            </span>
          )
        })}
      </div>

      <div className="mt-4 p-4 bg-slate-700/50 rounded-lg border border-slate-600">
        <p className="text-slate-400 text-sm">
          <span className="font-semibold text-slate-300">How to read:</span> Hover over
          any sentence to see its AI confidence score. Darker highlighting indicates
          stronger AI patterns detected.
        </p>
      </div>
    </div>
  )
}
