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

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
};

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
    <motion.div variants={item} className="glass-panel p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          <Settings2 className="w-5 h-5 text-primary-400" />
          Training Configuration
        </h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Model Settings */}
        <div className="space-y-4">
          <h3 className="font-medium text-white/80 flex items-center gap-2">
            <Layers className="w-4 h-4" />
            Model Architecture
          </h3>

          <div>
            <label className="text-sm text-white/60">Network Type</label>
            <select
              className="input-field mt-1"
              value={config.model_type}
              onChange={(e) => setConfig({ ...config, model_type: e.target.value as TrainingConfig['model_type'] })}
            >
              <option value="siren">SIREN</option>
              <option value="fourier">Fourier Features</option>
              <option value="modulated">Modulated SIREN</option>
            </select>
          </div>

          <div>
            <label className="text-sm text-white/60">Hidden Dimension</label>
            <input
              type="number"
              className="input-field mt-1"
              value={config.hidden_dim}
              onChange={(e) => setConfig({ ...config, hidden_dim: parseInt(e.target.value) })}
            />
          </div>

          <div>
            <label className="text-sm text-white/60">Number of Layers</label>
            <input
              type="number"
              className="input-field mt-1"
              value={config.num_layers}
              onChange={(e) => setConfig({ ...config, num_layers: parseInt(e.target.value) })}
            />
          </div>
        </div>

        {/* Physics Settings */}
        <div className="space-y-4">
          <h3 className="font-medium text-white/80 flex items-center gap-2">
            <Waves className="w-4 h-4" />
            Physics Configuration
          </h3>

          <div>
            <label className="text-sm text-white/60">Wave Equation</label>
            <select
              className="input-field mt-1"
              value={config.wave_equation}
              onChange={(e) => setConfig({ ...config, wave_equation: e.target.value as TrainingConfig['wave_equation'] })}
            >
              <option value="acoustic">Acoustic</option>
              <option value="elastic">Elastic</option>
              <option value="viscoacoustic">Viscoacoustic</option>
            </select>
          </div>

          <div>
            <label className="text-sm text-white/60">Velocity Model</label>
            <select
              className="input-field mt-1"
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

          <div>
            <label className="text-sm text-white/60">Source Frequency (Hz)</label>
            <input
              type="number"
              step="0.5"
              className="input-field mt-1"
              value={config.source_frequency}
              onChange={(e) => setConfig({ ...config, source_frequency: parseFloat(e.target.value) })}
            />
          </div>
        </div>

        {/* Training Settings */}
        <div className="space-y-4">
          <h3 className="font-medium text-white/80 flex items-center gap-2">
            <Zap className="w-4 h-4" />
            Training Parameters
          </h3>

          <div>
            <label className="text-sm text-white/60">Learning Rate</label>
            <input
              type="number"
              step="0.0001"
              className="input-field mt-1"
              value={config.learning_rate}
              onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })}
            />
          </div>

          <div>
            <label className="text-sm text-white/60">Batch Size</label>
            <input
              type="number"
              className="input-field mt-1"
              value={config.batch_size}
              onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
            />
          </div>

          <div>
            <label className="text-sm text-white/60">Epochs</label>
            <input
              type="number"
              className="input-field mt-1"
              value={config.num_epochs}
              onChange={(e) => setConfig({ ...config, num_epochs: parseInt(e.target.value) })}
            />
          </div>
        </div>
      </div>

      {/* Loss Weights */}
      <div className="border-t border-white/10 pt-6">
        <h3 className="font-medium text-white/80 mb-4">Loss Weights</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="text-sm text-white/60">Physics Weight</label>
            <input
              type="number"
              step="0.1"
              className="input-field mt-1"
              value={config.physics_weight}
              onChange={(e) => setConfig({ ...config, physics_weight: parseFloat(e.target.value) })}
            />
          </div>
          <div>
            <label className="text-sm text-white/60">Data Weight</label>
            <input
              type="number"
              step="0.1"
              className="input-field mt-1"
              value={config.data_weight}
              onChange={(e) => setConfig({ ...config, data_weight: parseFloat(e.target.value) })}
            />
          </div>
          <div>
            <label className="text-sm text-white/60">Boundary Weight</label>
            <input
              type="number"
              step="0.1"
              className="input-field mt-1"
              value={config.boundary_weight}
              onChange={(e) => setConfig({ ...config, boundary_weight: parseFloat(e.target.value) })}
            />
          </div>
          <div>
            <label className="text-sm text-white/60">Adaptive Weights</label>
            <select
              className="input-field mt-1"
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

      {/* Start Button */}
      <div className="flex justify-end">
        <button
          onClick={onStart}
          disabled={isLoading}
          className="btn-primary flex items-center gap-2"
        >
          {isLoading ? (
            <RefreshCw className="w-5 h-5 animate-spin" />
          ) : (
            <Play className="w-5 h-5" />
          )}
          Start Training
        </button>
      </div>
    </motion.div>
  );
}

function TrainingProgress() {
  const { progress, metricHistory, activeJob, wsConnected } = useTrainingStore();
  const pauseMutation = usePauseTraining();
  const resumeMutation = useResumeTraining();
  const cancelMutation = useCancelTraining();

  // Connect WebSocket when there's an active job
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
      variants={item}
      initial="hidden"
      animate="show"
      className="glass-panel p-6 space-y-6"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <TrendingDown className="w-5 h-5 text-primary-400" />
            Training Progress
          </h2>
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                wsConnected ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'
              }`}
            />
            <span className="text-sm text-white/60">
              {wsConnected ? 'Live' : 'Connecting...'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isRunning && (
            <button
              onClick={() => activeJob && pauseMutation.mutate(activeJob.job_id)}
              className="btn-secondary flex items-center gap-2"
            >
              <Pause className="w-4 h-4" />
              Pause
            </button>
          )}
          {isPaused && (
            <button
              onClick={() => activeJob && resumeMutation.mutate(activeJob.job_id)}
              className="btn-primary flex items-center gap-2"
            >
              <Play className="w-4 h-4" />
              Resume
            </button>
          )}
          <button
            onClick={() => activeJob && cancelMutation.mutate(activeJob.job_id)}
            className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-xl transition-all flex items-center gap-2"
          >
            <Square className="w-4 h-4" />
            Stop
          </button>
        </div>
      </div>

      {/* Progress Stats */}
      {progress && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 bg-white/5 rounded-xl">
            <p className="text-sm text-white/60">Epoch</p>
            <p className="text-2xl font-bold">
              {progress.epoch}/{progress.total_epochs}
            </p>
          </div>
          <div className="p-4 bg-white/5 rounded-xl">
            <p className="text-sm text-white/60">Total Loss</p>
            <p className="text-2xl font-bold">{progress.loss.toExponential(3)}</p>
          </div>
          <div className="p-4 bg-white/5 rounded-xl">
            <p className="text-sm text-white/60">Learning Rate</p>
            <p className="text-2xl font-bold">{progress.learning_rate.toExponential(2)}</p>
          </div>
          <div className="p-4 bg-white/5 rounded-xl">
            <p className="text-sm text-white/60">ETA</p>
            <p className="text-2xl font-bold flex items-center gap-2">
              <Clock className="w-5 h-5" />
              {Math.floor(progress.eta / 60)}m {Math.floor(progress.eta % 60)}s
            </p>
          </div>
        </div>
      )}

      {/* Progress Bar */}
      {progress && (
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-white/60">Overall Progress</span>
            <span>{((progress.epoch / progress.total_epochs) * 100).toFixed(1)}%</span>
          </div>
          <div className="h-3 bg-white/10 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-primary-500 to-accent-500"
              initial={{ width: 0 }}
              animate={{ width: `${(progress.epoch / progress.total_epochs) * 100}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>
      )}

      {/* Loss Chart */}
      {chartData.length > 0 && (
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.4)" fontSize={12} />
              <YAxis
                stroke="rgba(255,255,255,0.4)"
                fontSize={12}
                tickFormatter={(v) => v.toExponential(1)}
                scale="log"
                domain={['auto', 'auto']}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(15, 23, 42, 0.95)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '12px',
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="total"
                stroke="#0ea5e9"
                strokeWidth={2}
                dot={false}
                name="Total"
              />
              <Line
                type="monotone"
                dataKey="physics"
                stroke="#d946ef"
                strokeWidth={2}
                dot={false}
                name="Physics"
              />
              <Line
                type="monotone"
                dataKey="data"
                stroke="#22c55e"
                strokeWidth={2}
                dot={false}
                name="Data"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
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
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="space-y-8"
    >
      {/* Header */}
      <motion.div variants={item}>
        <h1 className="text-3xl font-bold">Training</h1>
        <p className="text-white/60 mt-1">
          Configure and monitor PINN training jobs
        </p>
      </motion.div>

      {/* Configuration Panel */}
      <ConfigPanel
        config={config}
        setConfig={setConfig}
        onStart={handleStart}
        isLoading={startMutation.isPending}
      />

      {/* Training Progress */}
      <AnimatePresence>
        <TrainingProgress />
      </AnimatePresence>
    </motion.div>
  );
}
