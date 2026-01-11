import { useState, useEffect, useCallback } from 'react'
import Header from './components/Header'
import LossChart from './components/LossChart'
import ThroughputGauge from './components/ThroughputGauge'
import VRAMGauge from './components/VRAMGauge'
import StatsCards from './components/StatsCards'
import MoDHeatmap from './components/MoDHeatmap'
import MoRDepthChart from './components/MoRDepthChart'
import GradNormChart from './components/GradNormChart'
import LRChart from './components/LRChart'
import RunSidebar from './components/RunSidebar'
import useTrainingData from './hooks/useTrainingData'

function App() {
  const [models, setModels] = useState([])
  const [runs, setRuns] = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedRun, setSelectedRun] = useState('')
  const [isLive, setIsLive] = useState(false)

  const { data, stats, runInfo, loading, error } = useTrainingData(selectedRun, isLive)

  // Fetch models on mount
  useEffect(() => {
    fetch('/api/models')
      .then(res => res.json())
      .then(data => {
        setModels(data)
        if (data.length > 0 && !selectedModel) {
          setSelectedModel(data[0].model_id)
        }
      })
      .catch(err => console.error('Failed to fetch models:', err))
  }, [])

  // Fetch runs when model changes
  useEffect(() => {
    if (!selectedModel) return

    fetch(`/api/runs/${selectedModel}`)
      .then(res => res.json())
      .then(data => {
        setRuns(data)
        if (data.length > 0) {
          // Select most recent run
          setSelectedRun(data[data.length - 1].run_id)
        }
      })
      .catch(err => console.error('Failed to fetch runs:', err))
  }, [selectedModel])

  const handleModelChange = useCallback((modelId) => {
    setSelectedModel(modelId)
    setSelectedRun('')
    setIsLive(false)
  }, [])

  const handleRunChange = useCallback((runId) => {
    setSelectedRun(runId)
    setIsLive(false)
  }, [])

  const toggleLive = useCallback(() => {
    setIsLive(prev => !prev)
  }, [])

  // Get latest metrics for gauges
  const latestStep = data.steps?.length > 0 ? data.steps[data.steps.length - 1] : null
  const avgTps = data.steps?.length > 0
    ? data.steps.slice(-50).reduce((sum, s) => sum + (s.tokens_per_sec || 0), 0) / Math.min(50, data.steps.length)
    : 0

  return (
    <div className="dashboard">
      <Header
        models={models}
        runs={runs}
        selectedModel={selectedModel}
        selectedRun={selectedRun}
        isLive={isLive}
        onModelChange={handleModelChange}
        onRunChange={handleRunChange}
        onToggleLive={toggleLive}
        currentStep={latestStep?.step}
        bestLoss={stats?.best_loss}
      />

      <main className="dashboard-main">
        <div className="dashboard-content">
          {error && (
            <div className="card" style={{ borderColor: '#ef4444' }}>
              <p style={{ color: '#ef4444' }}>Error: {error}</p>
            </div>
          )}

          {loading && !data.steps?.length ? (
            <div className="loading">Loading training data...</div>
          ) : !selectedRun ? (
            <div className="empty-state">
              <p>Select a model and run to view training metrics</p>
            </div>
          ) : (
            <>
              {/* Row 1: Core Metrics */}
              <div className="grid-row grid-row-3">
                <div className="card">
                  <div className="card-header">
                    <span className="card-title">Loss</span>
                  </div>
                  <div className="chart-container tall">
                    <LossChart data={data.steps || []} />
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <span className="card-title">Throughput</span>
                  </div>
                  <ThroughputGauge
                    current={latestStep?.tokens_per_sec || 0}
                    average={avgTps}
                    target={12000}
                  />
                </div>

                <div className="card">
                  <div className="card-header">
                    <span className="card-title">VRAM</span>
                  </div>
                  <VRAMGauge
                    current={latestStep?.vram_gb || 0}
                    max={32}
                  />
                </div>
              </div>

              {/* Stats Cards */}
              <StatsCards
                currentStep={latestStep?.step || 0}
                maxStep={data.maxStep || stats?.max_step || 0}
                bestLoss={stats?.best_loss}
                runInfo={runInfo}
              />

              {/* Row 2: Routing Health */}
              <div className="grid-row grid-row-2">
                <div className="card">
                  <div className="card-header">
                    <span className="card-title">MoD Capacity (per layer)</span>
                  </div>
                  <div className="chart-container tall">
                    <MoDHeatmap data={data.routingMod || []} />
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <span className="card-title">MoR Depth</span>
                  </div>
                  <div className="chart-container tall">
                    <MoRDepthChart data={data.routingMor || []} />
                  </div>
                </div>
              </div>

              {/* Row 3: Training Dynamics */}
              <div className="grid-row grid-row-2">
                <div className="card">
                  <div className="card-header">
                    <span className="card-title">Learning Rate</span>
                  </div>
                  <div className="chart-container">
                    <LRChart data={data.steps || []} />
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <span className="card-title">Gradient Norm</span>
                  </div>
                  <div className="chart-container">
                    <GradNormChart data={data.steps || []} />
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        <aside className="dashboard-sidebar">
          <RunSidebar
            runInfo={runInfo}
            stats={stats}
            modelId={selectedModel}
          />
        </aside>
      </main>
    </div>
  )
}

export default App
