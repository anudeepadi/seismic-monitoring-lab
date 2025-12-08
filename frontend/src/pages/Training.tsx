import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Play,
  Pause,
  Square,
  Settings2,
  Cpu,
  Layers,
  Waves,
  Zap,
  Clock,
  TrendingDown,
  RefreshCw,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { useStartTraining, usePauseTraining, useResumeTraining, useCancelTraining } from '../hooks/useApi';
import { useTrainingStore, defaultTrainingConfig } from '../stores/trainingStore';
import { useWebSocket } from '../hooks/useWebSocket';
import type { TrainingConfig } from '../types';

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

  .training-container {
    min-height: 100vh;
    background: #0a0a0a;
    color: #e5e5e5;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    padding: 0;
    margin: -32px;
  }

  .training-header {
    background: linear-gradient(180deg, #0d0d0d 0%, #0a0a0a 100%);
    border-bottom: 1px solid #1f1f1f;
    padding: 24px 32px;
  }

  .header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 24px;
    font-weight: 700;
    color: #ffffff;
  }

  .header-subtitle {
    font-size: 13px;
    color: #737373;
    margin-top: 4px;
  }

  .training-content {
    padding: 24px 32px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .panel {
    background: #0d0d0d;
    border: 1px solid #1f1f1f;
    border-radius: 8px;
    overflow: hidden;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px;
    border-bottom: 1px solid #1f1f1f;
    background: #0a0a0a;
  }

  .panel-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    color: #e5e5e5;
  }

  .panel-title svg {
    width: 18px;
    height: 18px;
    color: #3b82f6;
  }

  .panel-body {
    padding: 24px;
  }

  .config-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 32px;
  }

  .config-section h3 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: #a3a3a3;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .config-section h3 svg {
    width: 16px;
    height: 16px;
    color: #60a5fa;
  }

  .form-group {
    margin-bottom: 16px;
  }

  .form-label {
    display: block;
    font-size: 12px;
    color: #737373;
    margin-bottom: 6px;
  }

  .form-input, .form-select {
    width: 100%;
    padding: 10px 12px;
    background: #141414;
    border: 1px solid #262626;
    border-radius: 6px;
    color: #e5e5e5;
    font-size: 13px;
    font-family: 'IBM Plex Mono', monospace;
    transition: all 0.15s ease;
  }

  .form-input:focus, .form-select:focus {
    outline: none;
    border-color: #3b82f6;
    background: #1a1a1a;
  }

  .form-select {
    cursor: pointer;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%23737373'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center;
    background-size: 16px;
    padding-right: 36px;
  }

  .weights-section {
    border-top: 1px solid #1f1f1f;
    padding-top: 24px;
    margin-top: 8px;
  }

  .weights-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
  }

  .panel-footer {
    padding: 16px 24px;
    border-top: 1px solid #1f1f1f;
    display: flex;
    justify-content: flex-end;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    border: none;
  }

  .btn-primary {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
  }

  .btn-primary:hover {
    background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
    transform: translateY(-1px);
  }

  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }

  .btn-secondary {
    background: #262626;
    color: #e5e5e5;
    border: 1px solid #333;
  }

  .btn-secondary:hover {
    background: #333;
  }

  .btn-danger {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
  }

  .btn-danger:hover {
    background: rgba(239, 68, 68, 0.25);
  }

  .status-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulse 2s infinite;
  }

  .status-dot.online { background: #22c55e; }
  .status-dot.connecting { background: #fbbf24; }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .status-label {
    font-size: 12px;
    color: #737373;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }

  .stat-card {
    background: #141414;
    border: 1px solid #1f1f1f;
    border-radius: 6px;
    padding: 16px;
  }

  .stat-label {
    font-size: 11px;
    color: #525252;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 24px;
    font-weight: 700;
    color: #e5e5e5;
    margin-top: 4px;
  }

  .progress-section {
    margin-bottom: 24px;
  }

  .progress-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 12px;
    color: #737373;
  }

  .progress-bar {
    height: 8px;
    background: #1f1f1f;
    border-radius: 4px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    border-radius: 4px;
    transition: width 0.3s ease;
  }

  .chart-container {
    height: 280px;
  }

  .controls-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  @media (max-width: 1200px) {
    .config-grid {
      grid-template-columns: repeat(2, 1fr);
    }
    .weights-grid {
      grid-template-columns: repeat(2, 1fr);
    }
    .stats-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  @media (max-width: 768px) {
    .config-grid {
      grid-template-columns: 1fr;
    }
    .weights-grid {
      grid-template-columns: 1fr;
    }
  }
`;

function ConfigPanel({
  config,
  setConfig,
  onStart,
  isLoading,
}: {
  config: TrainingConfig;
  setConfig: (config: TrainingConfig) => void;
  onStart: () => void;
  isLoading: boolean;
}) {
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <Settings2 />
          Training Configuration
        </div>
      </div>
      <div className="panel-body">
        <div className="config-grid">
          {/* Model Settings */}
          <div className="config-section">
            <h3><Layers size={16} /> Model Architecture</h3>
            <div className="form-group">
              <label className="form-label">Network Type</label>
              <select
                className="form-select"
                value={config.model_type}
                onChange={(e) => setConfig({ ...config, model_type: e.target.value as TrainingConfig['model_type'] })}
              >
                <option value="siren">SIREN</option>
                <option value="fourier">Fourier Features</option>
                <option value="modulated">Modulated SIREN</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Hidden Dimension</label>
              <input
                type="number"
                className="form-input"
                value={config.hidden_dim}
                onChange={(e) => setConfig({ ...config, hidden_dim: parseInt(e.target.value) })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Number of Layers</label>
              <input
                type="number"
                className="form-input"
                value={config.num_layers}
                onChange={(e) => setConfig({ ...config, num_layers: parseInt(e.target.value) })}
              />
            </div>
          </div>

          {/* Physics Settings */}
          <div className="config-section">
            <h3><Waves size={16} /> Physics Configuration</h3>
            <div className="form-group">
              <label className="form-label">Wave Equation</label>
              <select
                className="form-select"
                value={config.wave_equation}
                onChange={(e) => setConfig({ ...config, wave_equation: e.target.value as TrainingConfig['wave_equation'] })}
              >
                <option value="acoustic">Acoustic</option>
                <option value="elastic">Elastic</option>
                <option value="viscoacoustic">Viscoacoustic</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Velocity Model</label>
              <select
                className="form-select"
                value={config.velocity_model}
                onChange={(e) => setConfig({ ...config, velocity_model: e.target.value as TrainingConfig['velocity_model'] })}
              >
                <option value="marmousi">Marmousi</option>
                <option value="layered">Layered</option>
                <option value="random">Random Gaussian</option>
                <option value="salt_dome">Salt Dome</option>
                <option value="fault">Fault</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Source Frequency (Hz)</label>
              <input
                type="number"
                step="0.5"
                className="form-input"
                value={config.source_frequency}
                onChange={(e) => setConfig({ ...config, source_frequency: parseFloat(e.target.value) })}
              />
            </div>
          </div>

          {/* Training Settings */}
          <div className="config-section">
            <h3><Zap size={16} /> Training Parameters</h3>
            <div className="form-group">
              <label className="form-label">Learning Rate</label>
              <input
                type="number"
                step="0.0001"
                className="form-input"
                value={config.learning_rate}
                onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Batch Size</label>
              <input
                type="number"
                className="form-input"
                value={config.batch_size}
                onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Epochs</label>
              <input
                type="number"
                className="form-input"
                value={config.num_epochs}
                onChange={(e) => setConfig({ ...config, num_epochs: parseInt(e.target.value) })}
              />
            </div>
          </div>
        </div>

        {/* Loss Weights */}
        <div className="weights-section">
          <h3 style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '12px', fontWeight: 600, color: '#a3a3a3', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '16px' }}>
            Loss Weights
          </h3>
          <div className="weights-grid">
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Physics Weight</label>
              <input
                type="number"
                step="0.1"
                className="form-input"
                value={config.physics_weight}
                onChange={(e) => setConfig({ ...config, physics_weight: parseFloat(e.target.value) })}
              />
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Data Weight</label>
              <input
                type="number"
                step="0.1"
                className="form-input"
                value={config.data_weight}
                onChange={(e) => setConfig({ ...config, data_weight: parseFloat(e.target.value) })}
              />
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Boundary Weight</label>
              <input
                type="number"
                step="0.1"
                className="form-input"
                value={config.boundary_weight}
                onChange={(e) => setConfig({ ...config, boundary_weight: parseFloat(e.target.value) })}
              />
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Adaptive Weights</label>
              <select
                className="form-select"
                value={config.use_adaptive_weights ? config.adaptive_method : 'none'}
                onChange={(e) => {
                  if (e.target.value === 'none') {
                    setConfig({ ...config, use_adaptive_weights: false });
                  } else {
                    setConfig({
                      ...config,
                      use_adaptive_weights: true,
                      adaptive_method: e.target.value as TrainingConfig['adaptive_method'],
                    });
                  }
                }}
              >
                <option value="none">Disabled</option>
                <option value="gradnorm">GradNorm</option>
                <option value="uncertainty">Uncertainty</option>
                <option value="softadapt">SoftAdapt</option>
              </select>
            </div>
          </div>
        </div>
      </div>
      <div className="panel-footer">
        <button onClick={onStart} disabled={isLoading} className="btn btn-primary">
          {isLoading ? <RefreshCw size={16} className="animate-spin" /> : <Play size={16} />}
          Start Training
        </button>
      </div>
    </div>
  );
}

function TrainingProgress() {
  const { progress, metricHistory, activeJob, wsConnected } = useTrainingStore();
  const pauseMutation = usePauseTraining();
  const resumeMutation = useResumeTraining();
  const cancelMutation = useCancelTraining();

  useWebSocket({ jobId: activeJob?.job_id || null });

  if (!activeJob && !progress) {
    return null;
  }

  const isPaused = activeJob?.status === 'paused';
  const isRunning = activeJob?.status === 'running';

  const chartData = metricHistory.slice(-100).map((m) => ({
    epoch: m.epoch,
    total: m.loss,
    physics: m.physics_loss,
    data: m.data_loss,
    boundary: m.boundary_loss,
  }));

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="panel"
    >
      <div className="panel-header">
        <div className="panel-title">
          <TrendingDown />
          Training Progress
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
          <div className="status-indicator">
            <span className={`status-dot ${wsConnected ? 'online' : 'connecting'}`} />
            <span className="status-label">{wsConnected ? 'Live' : 'Connecting...'}</span>
          </div>
          <div className="controls-row">
            {isRunning && (
              <button
                onClick={() => activeJob && pauseMutation.mutate(activeJob.job_id)}
                className="btn btn-secondary"
              >
                <Pause size={14} />
                Pause
              </button>
            )}
            {isPaused && (
              <button
                onClick={() => activeJob && resumeMutation.mutate(activeJob.job_id)}
                className="btn btn-primary"
              >
                <Play size={14} />
                Resume
              </button>
            )}
            <button
              onClick={() => activeJob && cancelMutation.mutate(activeJob.job_id)}
              className="btn btn-danger"
            >
              <Square size={14} />
              Stop
            </button>
          </div>
        </div>
      </div>
      <div className="panel-body">
        {progress && (
          <>
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-label">Epoch</div>
                <div className="stat-value">{progress.epoch}/{progress.total_epochs}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Total Loss</div>
                <div className="stat-value">{progress.loss.toExponential(3)}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Learning Rate</div>
                <div className="stat-value">{progress.learning_rate.toExponential(2)}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">ETA</div>
                <div className="stat-value" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Clock size={20} />
                  {Math.floor(progress.eta / 60)}m {Math.floor(progress.eta % 60)}s
                </div>
              </div>
            </div>

            <div className="progress-section">
              <div className="progress-header">
                <span>Overall Progress</span>
                <span>{((progress.epoch / progress.total_epochs) * 100).toFixed(1)}%</span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${(progress.epoch / progress.total_epochs) * 100}%` }}
                />
              </div>
            </div>
          </>
        )}

        {chartData.length > 0 && (
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" />
                <XAxis dataKey="epoch" stroke="#525252" fontSize={11} />
                <YAxis
                  stroke="#525252"
                  fontSize={11}
                  tickFormatter={(v) => v.toExponential(1)}
                  scale="log"
                  domain={['auto', 'auto']}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '6px',
                    fontFamily: 'IBM Plex Mono, monospace',
                    fontSize: '12px',
                  }}
                />
                <Legend />
                <Line type="monotone" dataKey="total" stroke="#3b82f6" strokeWidth={2} dot={false} name="Total" />
                <Line type="monotone" dataKey="physics" stroke="#8b5cf6" strokeWidth={2} dot={false} name="Physics" />
                <Line type="monotone" dataKey="data" stroke="#22c55e" strokeWidth={2} dot={false} name="Data" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </motion.div>
  );
}

export default function Training() {
  const [config, setConfig] = useState<TrainingConfig>(defaultTrainingConfig);
  const { setActiveJob, clearMetricHistory } = useTrainingStore();
  const startMutation = useStartTraining();

  const handleStart = async () => {
    clearMetricHistory();
    try {
      const result = await startMutation.mutateAsync(config);
      setActiveJob({
        job_id: result.job_id,
        status: 'running',
        config,
        progress: null,
        created_at: new Date().toISOString(),
        started_at: new Date().toISOString(),
        completed_at: null,
        error: null,
      });
    } catch (error) {
      console.error('Failed to start training:', error);
    }
  };

  return (
    <>
      <style>{styles}</style>
      <div className="training-container">
        <header className="training-header">
          <h1 className="header-title">TRAINING</h1>
          <p className="header-subtitle">Configure and monitor PINN training jobs</p>
        </header>

        <div className="training-content">
          <ConfigPanel
            config={config}
            setConfig={setConfig}
            onStart={handleStart}
            isLoading={startMutation.isPending}
          />
          <AnimatePresence>
            <TrainingProgress />
          </AnimatePresence>
        </div>
      </div>
    </>
  );
}
