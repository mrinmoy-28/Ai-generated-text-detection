import React, { useState, useMemo } from 'react'

export default function HistoryTable({ historyData }) {
  const [currentPage, setCurrentPage] = useState(1)
  const [filterVerdict, setFilterVerdict] = useState('all')
  const itemsPerPage = 20

  const filteredData = useMemo(() => {
    if (filterVerdict === 'all') {
      return historyData
    }
    return historyData.filter(item => item.verdict === filterVerdict)
  }, [historyData, filterVerdict])

  const totalPages = Math.ceil(filteredData.length / itemsPerPage)
  const startIndex = (currentPage - 1) * itemsPerPage
  const paginatedData = filteredData.slice(startIndex, startIndex + itemsPerPage)

  const getVerdictColor = (verdict) => {
    return verdict === 'AI Generated' ? 'bg-red-500/20 text-red-300' : 'bg-green-500/20 text-green-300'
  }

  const getSourceColor = (source) => {
    const colors = {
      text: 'bg-blue-500/20 text-blue-300',
      file: 'bg-purple-500/20 text-purple-300',
      batch: 'bg-indigo-500/20 text-indigo-300',
    }
    return colors[source] || colors.text
  }

  const getConfidenceColor = (verdict, confidence) => {
    const isAI = verdict === 'AI Generated'
    if (confidence >= 70) {
      return isAI ? 'text-red-400' : 'text-green-400'
    }
    if (confidence >= 40) {
      return 'text-yellow-400'
    }
    return isAI ? 'text-green-400' : 'text-red-400'
  }

  const truncateText = (text, length = 50) => {
    return text.length > length ? text.substring(0, length) + '...' : text
  }

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700">
      <div className="p-6 border-b border-slate-700">
        <h2 className="text-2xl font-bold text-white mb-4">Submission History</h2>

        <div className="flex gap-3">
          {['all', 'AI Generated', 'Human Written'].map((verdict) => (
            <button
              key={verdict}
              onClick={() => {
                setFilterVerdict(verdict)
                setCurrentPage(1)
              }}
              className={`px-4 py-2 rounded-lg font-medium transition ${
                filterVerdict === verdict
                  ? 'bg-primary text-white'
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              {verdict === 'all' ? 'All Submissions' : verdict}
            </button>
          ))}
        </div>
      </div>

      {paginatedData.length === 0 ? (
        <div className="p-12 text-center">
          <p className="text-slate-400 text-lg">No submissions yet</p>
        </div>
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-700/50 border-b border-slate-600">
                <tr className="text-left text-slate-300 text-sm font-semibold">
                  <th className="px-6 py-3">#</th>
                  <th className="px-6 py-3">Text Preview</th>
                  <th className="px-6 py-3">Verdict</th>
                  <th className="px-6 py-3">Confidence</th>
                  <th className="px-6 py-3">Source</th>
                  <th className="px-6 py-3">Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {paginatedData.map((item, index) => (
                  <tr
                    key={item.id}
                    className="border-b border-slate-700 hover:bg-slate-700/30 transition"
                  >
                    <td className="px-6 py-4 text-slate-300 font-medium">
                      {startIndex + index + 1}
                    </td>
                    <td className="px-6 py-4">
                      <p className="text-slate-300 text-sm max-w-xs">
                        {truncateText(item.text)}
                      </p>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${getVerdictColor(item.verdict)}`}>
                        {item.verdict}
                      </span>
                    </td>
                    <td className={`px-6 py-4 font-semibold ${getConfidenceColor(item.verdict, item.confidence)}`}>
                      {item.confidence.toFixed(1)}%
                    </td>
                    <td className="px-6 py-4">
                      <span className={`px-3 py-1 rounded-full text-sm font-medium capitalize ${getSourceColor(item.source)}`}>
                        {item.source}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-slate-400 text-sm">
                      {new Date(item.timestamp).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {totalPages > 1 && (
            <div className="flex justify-center items-center gap-2 p-6 border-t border-slate-700">
              <button
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded transition"
              >
                ← Previous
              </button>

              <div className="flex gap-1">
                {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
                  <button
                    key={page}
                    onClick={() => setCurrentPage(page)}
                    className={`px-3 py-1 rounded transition ${
                      currentPage === page
                        ? 'bg-primary text-white font-semibold'
                        : 'bg-slate-700 hover:bg-slate-600 text-white'
                    }`}
                  >
                    {page}
                  </button>
                ))}
              </div>

              <button
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded transition"
              >
                Next →
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
