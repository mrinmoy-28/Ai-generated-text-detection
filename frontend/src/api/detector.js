import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const checkHealth = async () => {
  try {
    const response = await api.get('/health')
    return response.data
  } catch (error) {
    throw error
  }
}

export const detectText = async (text) => {
  try {
    const response = await api.post('/detect', { text })
    return response.data
  } catch (error) {
    throw error
  }
}

export const detectSentences = async (text) => {
  try {
    const response = await api.post('/detect/sentences', { text })
    return response.data
  } catch (error) {
    throw error
  }
}

export const detectFile = async (file) => {
  try {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post('/detect/file', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  } catch (error) {
    throw error
  }
}

export const detectBatch = async (files) => {
  try {
    const formData = new FormData()
    files.forEach((file) => {
      formData.append('files', file)
    })
    const response = await api.post('/detect/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  } catch (error) {
    throw error
  }
}

export const explainText = async (text) => {
  try {
    const response = await api.post('/explain', { text })
    return response.data
  } catch (error) {
    throw error
  }
}

export const downloadReport = async (data) => {
  try {
    const response = await api.post('/report', data, {
      responseType: 'blob',
    })
    const url = window.URL.createObjectURL(new Blob([response.data]))
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', 'detection_report.pdf')
    document.body.appendChild(link)
    link.click()
    link.parentNode.removeChild(link)
    window.URL.revokeObjectURL(url)
  } catch (error) {
    throw error
  }
}

export const getHistory = async () => {
  try {
    const response = await api.get('/history')
    return response.data
  } catch (error) {
    throw error
  }
}

export const getStats = async () => {
  try {
    const response = await api.get('/stats')
    return response.data
  } catch (error) {
    throw error
  }
}

// New Production Features

export const compareDocuments = async (text1, text2) => {
  try {
    const response = await api.post('/compare', {
      text1,
      text2,
    })
    return response.data
  } catch (error) {
    throw error
  }
}

export const compareFiles = async (file1, file2) => {
  try {
    const formData = new FormData()
    formData.append('file1', file1)
    formData.append('file2', file2)
    const response = await api.post('/compare/files', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  } catch (error) {
    throw error
  }
}

export const analyzeAdvanced = async (text) => {
  try {
    const response = await api.post('/analyze/advanced', { text })
    return response.data
  } catch (error) {
    throw error
  }
}

export const extractText = async (file) => {
  try {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post('/extract-text', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  } catch (error) {
    throw error
  }
}

export const exportResults = async (results, format = 'pdf') => {
  try {
    const response = await api.post(
      '/export',
      { results, format },
      { responseType: 'blob' }
    )
    const url = window.URL.createObjectURL(new Blob([response.data]))
    const link = document.createElement('a')
    link.href = url
    const extension = format === 'pdf' ? 'pdf' : 'docx'
    link.setAttribute('download', `detection_results.${extension}`)
    document.body.appendChild(link)
    link.click()
    link.parentNode.removeChild(link)
    window.URL.revokeObjectURL(url)
  } catch (error) {
    throw error
  }
}

export const analyzePlagiarism = async (text) => {
  try {
    const response = await api.post('/plagiarism', { text })
    return response.data
  } catch (error) {
    throw error
  }
}

export const getProcessingStatus = async (jobId) => {
  try {
    const response = await api.get(`/status/${jobId}`)
    return response.data
  } catch (error) {
    throw error
  }
}
