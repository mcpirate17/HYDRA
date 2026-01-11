import { memo, useMemo } from 'react'
import { theme } from '../styles/theme'

function RunSidebar({ runInfo, stats, modelId }) {
  // Parse config from runInfo
  const config = useMemo(() => {
    if (!runInfo?.config) return {}
    return typeof runInfo.config === 'string'
      ? JSON.parse(runInfo.config)
      : runInfo.config
  }, [runInfo])

  return (
    <div>
      {/* Model Info */}
      <section className="sidebar-section">
        <h3 className="sidebar-section-title">Model</h3>

        <div className="sidebar-item">
          <span className="sidebar-item-label">Model ID</span>
          <span className="sidebar-item-value">{modelId || '—'}</span>
        </div>

        {config.model_size && (
          <div className="sidebar-item">
            <span className="sidebar-item-label">Size</span>
            <span className="sidebar-item-value">{config.model_size}</span>
          </div>
        )}

        {config.dim && (
          <div className="sidebar-item">
            <span className="sidebar-item-label">Dimension</span>
            <span className="sidebar-item-value">{config.dim}</span>
          </div>
        )}

        {config.n_blocks && (
          <div className="sidebar-item">
            <span className="sidebar-item-label">Blocks</span>
            <span className="sidebar-item-value">{config.n_blocks}</span>
          </div>
        )}

        {config.n_recursions && (
          <div className="sidebar-item">
            <span className="sidebar-item-label">Recursions</span>
            <span className="sidebar-item-value">{config.n_recursions}</span>
          </div>
        )}
      </section>

      {/* Run Info */}
      <section className="sidebar-section">
        <h3 className="sidebar-section-title">Run</h3>

        <div className="sidebar-item">
          <span className="sidebar-item-label">Run ID</span>
          <span className="sidebar-item-value" style={{ fontSize: 11 }}>
            {runInfo?.run_id || '—'}
          </span>
        </div>

        {runInfo?.start_time && (
          <div className="sidebar-item">
            <span className="sidebar-item-label">Started</span>
            <span className="sidebar-item-value">
              {new Date(runInfo.start_time).toLocaleString()}
            </span>
          </div>
        )}

        {stats?.best_loss != null && (
          <div className="sidebar-item">
            <span className="sidebar-item-label">Best Loss</span>
            <span className="sidebar-item-value" style={{ color: theme.status.success }}>
              {stats.best_loss.toFixed(4)}
            </span>
          </div>
        )}

        {stats?.step_count != null && (
          <div className="sidebar-item">
            <span className="sidebar-item-label">Total Steps</span>
            <span className="sidebar-item-value">
              {stats.step_count.toLocaleString()}
            </span>
          </div>
        )}
      </section>

      {/* Training Config */}
      <section className="sidebar-section">
        <h3 className="sidebar-section-title">Config</h3>

        {config.max_lr && (
          <div className="sidebar-item">
            <span className="sidebar-item-label">Max LR</span>
            <span className="sidebar-item-value">{config.max_lr}</span>
          </div>
        )}

        {config.batch_size && (
          <div className="sidebar-item">
            <span className="sidebar-item-label">Batch Size</span>
            <span className="sidebar-item-value">{config.batch_size}</span>
          </div>
        )}

        {config.grad_accum && (
          <div className="sidebar-item">
            <span className="sidebar-item-label">Grad Accum</span>
            <span className="sidebar-item-value">{config.grad_accum}</span>
          </div>
        )}

        {config.seq_len && (
          <div className="sidebar-item">
            <span className="sidebar-item-label">Seq Length</span>
            <span className="sidebar-item-value">{config.seq_len}</span>
          </div>
        )}

        {config.dataset_name && (
          <div className="sidebar-item">
            <span className="sidebar-item-label">Dataset</span>
            <span className="sidebar-item-value" style={{ fontSize: 11 }}>
              {config.dataset_name}
            </span>
          </div>
        )}
      </section>

      {/* Routing Config */}
      {(config.mod_capacity || config.mor_adaptive) && (
        <section className="sidebar-section">
          <h3 className="sidebar-section-title">Routing</h3>

          {config.mod_capacity && (
            <div className="sidebar-item">
              <span className="sidebar-item-label">MoD Capacity</span>
              <span className="sidebar-item-value">{config.mod_capacity}</span>
            </div>
          )}

          {config.mor_adaptive !== undefined && (
            <div className="sidebar-item">
              <span className="sidebar-item-label">MoR Adaptive</span>
              <span className="sidebar-item-value">
                {config.mor_adaptive ? 'Yes' : 'No'}
              </span>
            </div>
          )}

          {config.moe_enabled && (
            <div className="sidebar-item">
              <span className="sidebar-item-label">MoE Experts</span>
              <span className="sidebar-item-value">
                {config.moe_num_experts || 4}
              </span>
            </div>
          )}
        </section>
      )}

      {/* Stats Summary */}
      {stats && (
        <section className="sidebar-section">
          <h3 className="sidebar-section-title">Database Stats</h3>

          <div className="sidebar-item">
            <span className="sidebar-item-label">Min Step</span>
            <span className="sidebar-item-value">
              {stats.min_step?.toLocaleString() || '—'}
            </span>
          </div>

          <div className="sidebar-item">
            <span className="sidebar-item-label">Max Step</span>
            <span className="sidebar-item-value">
              {stats.max_step?.toLocaleString() || '—'}
            </span>
          </div>

          <div className="sidebar-item">
            <span className="sidebar-item-label">Avg Loss</span>
            <span className="sidebar-item-value">
              {stats.avg_loss?.toFixed(4) || '—'}
            </span>
          </div>

          <div className="sidebar-item">
            <span className="sidebar-item-label">Run Count</span>
            <span className="sidebar-item-value">
              {stats.run_count || '—'}
            </span>
          </div>
        </section>
      )}
    </div>
  )
}

export default memo(RunSidebar)
