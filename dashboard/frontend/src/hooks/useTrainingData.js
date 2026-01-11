import { useState, useEffect, useRef, useCallback } from 'react'

/**
 * Hook for fetching training data via REST and WebSocket.
 *
 * @param {string} runId - The run ID to fetch data for
 * @param {boolean} isLive - Whether to enable WebSocket updates
 */
function useTrainingData(runId, isLive = false) {
  const [data, setData] = useState({
    steps: [],
    routingMod: [],
    routingMor: [],
    maxStep: 0,
  })
  const [stats, setStats] = useState(null)
  const [runInfo, setRunInfo] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const wsRef = useRef(null)
  const lastStepRef = useRef(0)

  // Fetch initial data via REST
  const fetchData = useCallback(async () => {
    if (!runId) return

    setLoading(true)
    setError(null)

    try {
      // Fetch metrics, routing data, and run info in parallel
      const [metricsRes, modRes, morRes, runRes] = await Promise.all([
        fetch(`/api/metrics/${runId}?limit=2000`),
        fetch(`/api/routing/mod/${runId}?limit=1000`),
        fetch(`/api/routing/mor/${runId}?limit=1000`),
        fetch(`/api/run/${runId}`),
      ])

      if (!metricsRes.ok) throw new Error('Failed to fetch metrics')

      const [metrics, modData, morData, runData] = await Promise.all([
        metricsRes.json(),
        modRes.json(),
        morRes.json(),
        runRes.json(),
      ])

      setData({
        steps: metrics.steps || [],
        routingMod: modData.data || [],
        routingMor: morData.data || [],
        maxStep: metrics.max_step || 0,
      })

      lastStepRef.current = metrics.max_step || 0

      setRunInfo(runData)

      // Fetch model stats
      const modelId = runId.split('_')[0]
      const statsRes = await fetch(`/api/stats/${modelId}`)
      if (statsRes.ok) {
        const statsData = await statsRes.json()
        setStats(statsData)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [runId])

  // Initial fetch when runId changes
  useEffect(() => {
    fetchData()
  }, [fetchData])

  // WebSocket connection for live updates
  useEffect(() => {
    if (!runId || !isLive) {
      // Close existing connection
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
      return
    }

    // Create WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/${runId}`

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WebSocket connected for run:', runId)
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)

        if (message.type === 'update' && message.steps?.length > 0) {
          setData(prev => ({
            ...prev,
            steps: [...prev.steps, ...message.steps],
            routingMod: [...prev.routingMod, ...(message.routing_mod || [])],
            routingMor: [...prev.routingMor, ...(message.routing_mor || [])],
            maxStep: message.last_step,
          }))

          lastStepRef.current = message.last_step
        }
      } catch (err) {
        console.error('WebSocket message parse error:', err)
      }
    }

    ws.onerror = (err) => {
      console.error('WebSocket error:', err)
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
    }

    // Cleanup
    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close()
      }
    }
  }, [runId, isLive])

  return {
    data,
    stats,
    runInfo,
    loading,
    error,
    refetch: fetchData,
  }
}

export default useTrainingData
