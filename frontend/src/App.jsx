import { useState, useCallback } from 'react'

const STEPS = [
  { id: 'upload', label: '1. Upload', icon: '📁' },
  { id: 'profile', label: '2. Profile', icon: '🔍' },
  { id: 'remediate', label: '3. Remediate', icon: '🔧' },
  { id: 'build', label: '4. Build', icon: '🏗️' },
  { id: 'visualize', label: '5. Visualize', icon: '📊' },
]

function App() {
  const [currentStep, setCurrentStep] = useState('upload')
  const [sessionId, setSessionId] = useState(null)
  const [completedSteps, setCompletedSteps] = useState([])
  const [uploadedFile, setUploadedFile] = useState(null)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState(null)
  const [dragActive, setDragActive] = useState(false)

  // Profile state
  const [isProfiling, setIsProfiling] = useState(false)
  const [profileError, setProfileError] = useState(null)
  const [profileHtml, setProfileHtml] = useState(null)
  const [profileScores, setProfileScores] = useState(null)

  const handleDrag = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      handleFileSelect(files[0])
    }
  }, [])

  const handleFileSelect = async (file) => {
    setError(null)

    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('Only CSV files are accepted. Please select a .csv file.')
      return
    }

    setIsUploading(true)
    setUploadedFile(null)
    setSessionId(null)
    setCompletedSteps([])
    // Reset profile state on new upload
    setProfileHtml(null)
    setProfileScores(null)
    setProfileError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Upload failed')
      }

      const data = await response.json()
      setSessionId(data.session_id)
      setUploadedFile(file.name)
      setCompletedSteps(['upload'])
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsUploading(false)
    }
  }

  const handleFileInput = (e) => {
    const file = e.target.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const runProfile = async () => {
    if (!sessionId) return

    setIsProfiling(true)
    setProfileError(null)

    try {
      const response = await fetch(`/api/profile/${sessionId}`)

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Profiling failed')
      }

      const data = await response.json()
      setProfileHtml(data.html_report)
      setProfileScores(data.scores)

      // Mark profile as completed
      if (!completedSteps.includes('profile')) {
        setCompletedSteps([...completedSteps, 'profile'])
      }
    } catch (err) {
      setProfileError(err.message)
    } finally {
      setIsProfiling(false)
    }
  }

  const canNavigateTo = (stepId) => {
    const stepIndex = STEPS.findIndex(s => s.id === stepId)

    // Can always go to upload
    if (stepId === 'upload') return true

    // Can go to completed steps or the next step after all completed
    const previousStep = STEPS[stepIndex - 1]
    return completedSteps.includes(previousStep?.id)
  }

  const handleTabClick = (stepId) => {
    if (canNavigateTo(stepId)) {
      setCurrentStep(stepId)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>D2I Pipeline</h1>
        <span className="subtitle">Data to Insights</span>
      </header>

      {/* Tab Navigation */}
      <nav className="tabs">
        {STEPS.map((step, index) => (
          <button
            key={step.id}
            className={`tab ${currentStep === step.id ? 'active' : ''} ${completedSteps.includes(step.id) ? 'completed' : ''} ${!canNavigateTo(step.id) ? 'disabled' : ''}`}
            onClick={() => handleTabClick(step.id)}
            disabled={!canNavigateTo(step.id)}
          >
            <span className="tab-icon">{step.icon}</span>
            <span className="tab-label">{step.label}</span>
            {completedSteps.includes(step.id) && <span className="check">✓</span>}
          </button>
        ))}
      </nav>

      {/* Content Area */}
      <main className={`content ${currentStep === 'profile' && profileHtml ? 'content-wide' : ''}`}>
        {currentStep === 'upload' && (
          <div className="upload-section">
            <h2>Upload Your Data</h2>
            <p className="description">
              Upload a CSV file to begin the data processing pipeline.
              Only one file can be processed at a time.
            </p>

            <div
              className={`dropzone ${dragActive ? 'drag-active' : ''} ${isUploading ? 'uploading' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              {isUploading ? (
                <div className="uploading-state">
                  <div className="spinner"></div>
                  <p>Uploading...</p>
                </div>
              ) : (
                <>
                  <div className="dropzone-icon">📄</div>
                  <p className="dropzone-text">
                    Drag and drop your CSV file here
                  </p>
                  <p className="dropzone-or">or</p>
                  <label className="file-input-label">
                    <input
                      type="file"
                      accept=".csv"
                      onChange={handleFileInput}
                      className="file-input"
                    />
                    Browse Files
                  </label>
                  <p className="file-hint">Only .csv files are accepted</p>
                </>
              )}
            </div>

            {error && (
              <div className="error-message">
                <span className="error-icon">⚠️</span>
                {error}
              </div>
            )}

            {uploadedFile && !error && (
              <div className="success-message">
                <span className="success-icon">✅</span>
                <div>
                  <strong>{uploadedFile}</strong> uploaded successfully!
                  <br />
                  <small>Session ID: {sessionId}</small>
                </div>
              </div>
            )}

            {uploadedFile && (
              <button
                className="next-button"
                onClick={() => setCurrentStep('profile')}
              >
                Continue to Profile →
              </button>
            )}
          </div>
        )}

        {currentStep === 'profile' && (
          <div className="profile-section">
            {!profileHtml ? (
              // Show run profile button when no report exists
              <div className="profile-start">
                <h2>Data Quality Profile</h2>
                <p className="description">
                  Analyze your data for quality issues, missing values, and potential problems.
                </p>
                <p className="file-info">
                  File: <strong>{uploadedFile}</strong>
                </p>

                {profileError && (
                  <div className="error-message">
                    <span className="error-icon">⚠️</span>
                    {profileError}
                  </div>
                )}

                <button
                  className="run-profile-button"
                  onClick={runProfile}
                  disabled={isProfiling}
                >
                  {isProfiling ? (
                    <>
                      <span className="button-spinner"></span>
                      Analyzing Data...
                    </>
                  ) : (
                    <>🔍 Run Profile</>
                  )}
                </button>
              </div>
            ) : (
              // Show the HTML report
              <div className="profile-report">
                <div className="report-header">
                  <h2>Profile Results</h2>
                  <div className="report-actions">
                    <button
                      className="rerun-button"
                      onClick={runProfile}
                      disabled={isProfiling}
                    >
                      {isProfiling ? 'Re-analyzing...' : '↻ Re-run Profile'}
                    </button>
                    <button
                      className="next-button-small"
                      onClick={() => setCurrentStep('remediate')}
                    >
                      Continue to Remediate →
                    </button>
                  </div>
                </div>

                {/* Embedded HTML Report */}
                <div
                  className="report-container"
                  dangerouslySetInnerHTML={{ __html: profileHtml }}
                />
              </div>
            )}
          </div>
        )}

        {currentStep === 'remediate' && (
          <div className="step-placeholder">
            <h2>🔧 Data Remediation</h2>
            <p>Apply automated fixes and review issues.</p>
            <p className="coming-soon">Coming in next iteration</p>
          </div>
        )}

        {currentStep === 'build' && (
          <div className="step-placeholder">
            <h2>🏗️ Build Database</h2>
            <p>Create Postgres tables from your data.</p>
            <p className="coming-soon">Coming in next iteration</p>
          </div>
        )}

        {currentStep === 'visualize' && (
          <div className="step-placeholder">
            <h2>📊 Visualize</h2>
            <p>View charts and insights from your data.</p>
            <p className="coming-soon">Coming in next iteration</p>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>D2I Pipeline Manager v1.0</p>
      </footer>
    </div>
  )
}

export default App
