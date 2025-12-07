import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  FlaskConical,
  Plus,
  Play,
  BarChart3,
  TrendingUp,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  Layers,
  X,
  GitCompare,
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
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from 'recharts';
import { useExperiments, useCreateExperiment, useModels } from '../hooks/useApi';
import type { Experiment, Model } from '../types';

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.05 },
  },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
};

// Mock experiments
const mockExperiments: Experiment[] = [
  {
    id: 'exp-1',
    name: 'Architecture Comparison',
    description: 'Comparing SIREN, Fourier, and Modulated networks on Marmousi',
    status: 'completed',
    models: ['model-1', 'model-2', 'model-3'],
    metrics: { mse: 0.00234, psnr: 35.2, ssim: 0.945 },
    created_at: '2024-01-15T08:00:00Z',
    updated_at: '2024-01-15T12:30:00Z',
  },
  {
    id: 'exp-2',
    name: 'Learning Rate Study',
    description: 'Testing different learning rate schedules',
    status: 'running',
    models: ['model-4'],
    metrics: { mse: 0.00312, psnr: 32.8, ssim: 0.912 },
    created_at: '2024-01-14T14:00:00Z',
    updated_at: '2024-01-14T18:45:00Z',
  },
  {
    id: 'exp-3',
    name: 'Salt Body Reconstruction',
    description: 'Testing PINN accuracy on complex salt dome structures',
    status: 'completed',
    models: ['model-3'],
    metrics: { mse: 0.00456, psnr: 30.1, ssim: 0.878 },
    created_at: '2024-01-13T10:00:00Z',
    updated_at: '2024-01-13T16:20:00Z',
  },
];

// Mock comparison data
const comparisonData = [
  { epoch: 0, siren: 0.5, fourier: 0.48, modulated: 0.52 },
  { epoch: 100, siren: 0.12, fourier: 0.15, modulated: 0.11 },
  { epoch: 200, siren: 0.05, fourier: 0.07, modulated: 0.045 },
  { epoch: 300, siren: 0.025, fourier: 0.035, modulated: 0.022 },
  { epoch: 400, siren: 0.012, fourier: 0.018, modulated: 0.01 },
  { epoch: 500, siren: 0.008, fourier: 0.012, modulated: 0.006 },
  { epoch: 600, siren: 0.005, fourier: 0.008, modulated: 0.004 },
  { epoch: 700, siren: 0.0035, fourier: 0.006, modulated: 0.003 },
  { epoch: 800, siren: 0.0028, fourier: 0.0045, modulated: 0.0024 },
  { epoch: 900, siren: 0.0024, fourier: 0.0038, modulated: 0.002 },
  { epoch: 1000, siren: 0.0021, fourier: 0.0032, modulated: 0.0018 },
];

const radarData = [
  { metric: 'Accuracy', siren: 85, fourier: 78, modulated: 92 },
  { metric: 'Speed', siren: 90, fourier: 75, modulated: 70 },
  { metric: 'Memory', siren: 95, fourier: 80, modulated: 75 },
  { metric: 'Physics', siren: 88, fourier: 82, modulated: 90 },
  { metric: 'Generalization', siren: 75, fourier: 88, modulated: 85 },
];

function StatusBadge({ status }: { status: Experiment['status'] }) {
  const config = {
    running: {
      icon: Loader2,
      className: 'status-running',
      animate: true,
    },
    completed: {
      icon: CheckCircle,
      className: 'status-completed',
      animate: false,
    },
    failed: {
      icon: XCircle,
      className: 'status-failed',
      animate: false,
    },
  };

  const { icon: Icon, className, animate } = config[status];

  return (
    <span className={className}>
      <Icon className={`w-3 h-3 ${animate ? 'animate-spin' : ''}`} />
      {status}
    </span>
  );
}

function ExperimentCard({
  experiment,
  onClick,
}: {
  experiment: Experiment;
  onClick: () => void;
}) {
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <motion.div
      variants={item}
      className="glass-panel p-6 hover:border-white/20 transition-all duration-300 cursor-pointer group"
      onClick={onClick}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-accent-500/20 to-accent-600/10 border border-accent-500/30 flex items-center justify-center">
            <FlaskConical className="w-6 h-6 text-accent-400" />
          </div>
          <div>
            <h3 className="font-semibold group-hover:text-primary-400 transition-colors">
              {experiment.name}
            </h3>
            <p className="text-sm text-white/50 line-clamp-1">
              {experiment.description}
            </p>
          </div>
        </div>
        <StatusBadge status={experiment.status} />
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="p-2 bg-white/5 rounded-lg text-center">
          <p className="text-xs text-white/40">MSE</p>
          <p className="font-mono text-sm">{experiment.metrics.mse?.toExponential(2) || 'N/A'}</p>
        </div>
        <div className="p-2 bg-white/5 rounded-lg text-center">
          <p className="text-xs text-white/40">PSNR</p>
          <p className="font-mono text-sm">{experiment.metrics.psnr?.toFixed(1) || 'N/A'} dB</p>
        </div>
        <div className="p-2 bg-white/5 rounded-lg text-center">
          <p className="text-xs text-white/40">SSIM</p>
          <p className="font-mono text-sm">{experiment.metrics.ssim?.toFixed(3) || 'N/A'}</p>
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between pt-4 border-t border-white/10">
        <span className="text-xs text-white/40 flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {formatDate(experiment.updated_at)}
        </span>
        <span className="text-xs text-white/60 flex items-center gap-1">
          <Layers className="w-3 h-3" />
          {experiment.models.length} model{experiment.models.length !== 1 ? 's' : ''}
        </span>
      </div>
    </motion.div>
  );
}

function NewExperimentModal({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [selectedModels, setSelectedModels] = useState<string[]>([]);

  const { data: models } = useModels();
  const createMutation = useCreateExperiment();

  const handleSubmit = async () => {
    if (!name || selectedModels.length === 0) return;

    try {
      await createMutation.mutateAsync({
        name,
        description,
        model_ids: selectedModels,
      });
      onClose();
    } catch (error) {
      console.error('Failed to create experiment:', error);
    }
  };

  if (!isOpen) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-8"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="glass-panel w-full max-w-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-6 border-b border-white/10 flex items-center justify-between">
          <h2 className="text-xl font-semibold">New Experiment</h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/10 rounded-lg transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-4">
          <div>
            <label className="text-sm text-white/60">Experiment Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Architecture Comparison"
              className="input-field mt-1"
            />
          </div>

          <div>
            <label className="text-sm text-white/60">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe the experiment goals..."
              className="input-field mt-1 min-h-[80px] resize-none"
            />
          </div>

          <div>
            <label className="text-sm text-white/60 mb-2 block">Select Models to Compare</label>
            <div className="space-y-2 max-h-[200px] overflow-y-auto">
              {(models || []).map((model: Model) => (
                <label
                  key={model.id}
                  className="flex items-center gap-3 p-3 bg-white/5 rounded-xl hover:bg-white/10 cursor-pointer transition-all"
                >
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(model.id)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedModels([...selectedModels, model.id]);
                      } else {
                        setSelectedModels(selectedModels.filter((id) => id !== model.id));
                      }
                    }}
                    className="w-4 h-4 rounded border-white/20"
                  />
                  <div>
                    <p className="font-medium">{model.name}</p>
                    <p className="text-xs text-white/40">{model.type.toUpperCase()}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="p-6 border-t border-white/10 flex justify-end gap-3">
          <button onClick={onClose} className="btn-secondary">
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={!name || selectedModels.length === 0}
            className="btn-primary flex items-center gap-2"
          >
            <Play className="w-4 h-4" />
            Create & Run
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}

function ComparisonChart() {
  return (
    <motion.div variants={item} className="glass-panel p-6">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <GitCompare className="w-5 h-5 text-primary-400" />
        Loss Comparison
      </h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={comparisonData}>
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
              dataKey="siren"
              stroke="#0ea5e9"
              strokeWidth={2}
              dot={false}
              name="SIREN"
            />
            <Line
              type="monotone"
              dataKey="fourier"
              stroke="#d946ef"
              strokeWidth={2}
              dot={false}
              name="Fourier"
            />
            <Line
              type="monotone"
              dataKey="modulated"
              stroke="#22c55e"
              strokeWidth={2}
              dot={false}
              name="Modulated"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}

function PerformanceRadar() {
  return (
    <motion.div variants={item} className="glass-panel p-6">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <BarChart3 className="w-5 h-5 text-primary-400" />
        Performance Metrics
      </h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={radarData}>
            <PolarGrid stroke="rgba(255,255,255,0.1)" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 12 }} />
            <PolarRadiusAxis
              tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10 }}
              domain={[0, 100]}
            />
            <Radar
              name="SIREN"
              dataKey="siren"
              stroke="#0ea5e9"
              fill="#0ea5e9"
              fillOpacity={0.2}
            />
            <Radar
              name="Fourier"
              dataKey="fourier"
              stroke="#d946ef"
              fill="#d946ef"
              fillOpacity={0.2}
            />
            <Radar
              name="Modulated"
              dataKey="modulated"
              stroke="#22c55e"
              fill="#22c55e"
              fillOpacity={0.2}
            />
            <Legend />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}

export default function Experiments() {
  const { data: apiExperiments } = useExperiments();
  const [showNewModal, setShowNewModal] = useState(false);
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null);

  // Use mock data if no API data
  const experiments = apiExperiments?.length ? apiExperiments : mockExperiments;

  const runningCount = experiments.filter((e) => e.status === 'running').length;
  const completedCount = experiments.filter((e) => e.status === 'completed').length;

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="space-y-6"
    >
      {/* Header */}
      <motion.div variants={item} className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Experiments</h1>
          <p className="text-white/60 mt-1">
            Compare and analyze model performance
          </p>
        </div>

        <button
          onClick={() => setShowNewModal(true)}
          className="btn-primary flex items-center gap-2"
        >
          <Plus className="w-5 h-5" />
          New Experiment
        </button>
      </motion.div>

      {/* Stats */}
      <motion.div variants={item} className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-panel p-6 flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-primary-500/20 flex items-center justify-center">
            <FlaskConical className="w-6 h-6 text-primary-400" />
          </div>
          <div>
            <p className="text-2xl font-bold">{experiments.length}</p>
            <p className="text-sm text-white/60">Total Experiments</p>
          </div>
        </div>

        <div className="glass-panel p-6 flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-green-500/20 flex items-center justify-center">
            <Play className="w-6 h-6 text-green-400" />
          </div>
          <div>
            <p className="text-2xl font-bold">{runningCount}</p>
            <p className="text-sm text-white/60">Running</p>
          </div>
        </div>

        <div className="glass-panel p-6 flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-accent-500/20 flex items-center justify-center">
            <CheckCircle className="w-6 h-6 text-accent-400" />
          </div>
          <div>
            <p className="text-2xl font-bold">{completedCount}</p>
            <p className="text-sm text-white/60">Completed</p>
          </div>
        </div>
      </motion.div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ComparisonChart />
        <PerformanceRadar />
      </div>

      {/* Experiments Grid */}
      <motion.div variants={item}>
        <h2 className="text-xl font-semibold mb-4">All Experiments</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {experiments.map((experiment) => (
            <ExperimentCard
              key={experiment.id}
              experiment={experiment}
              onClick={() => setSelectedExperiment(experiment)}
            />
          ))}
        </div>
      </motion.div>

      {/* New Experiment Modal */}
      <AnimatePresence>
        {showNewModal && (
          <NewExperimentModal
            isOpen={showNewModal}
            onClose={() => setShowNewModal(false)}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
}
