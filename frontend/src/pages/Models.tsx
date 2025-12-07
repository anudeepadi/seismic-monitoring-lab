import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Box,
  Download,
  Trash2,
  Play,
  Info,
  Layers,
  Cpu,
  Clock,
  TrendingDown,
  Search,
  Filter,
  MoreVertical,
  X,
} from 'lucide-react';
import { useModels, useDeleteModel } from '../hooks/useApi';
import type { Model } from '../types';

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

// Mock data for demonstration
const mockModels: Model[] = [
  {
    id: 'model-1',
    name: 'SIREN Marmousi v1',
    type: 'siren',
    architecture: { hidden_dim: 256, num_layers: 6, total_params: 524288 },
    training: { epochs: 1000, final_loss: 0.00234, physics_loss: 0.00156, data_loss: 0.00078 },
    velocity_model: 'marmousi',
    created_at: '2024-01-15T10:30:00Z',
    checkpoint_path: '/checkpoints/model-1.pt',
  },
  {
    id: 'model-2',
    name: 'Fourier Features Layered',
    type: 'fourier',
    architecture: { hidden_dim: 512, num_layers: 8, total_params: 2097152 },
    training: { epochs: 2000, final_loss: 0.00189, physics_loss: 0.00098, data_loss: 0.00091 },
    velocity_model: 'layered',
    created_at: '2024-01-14T14:20:00Z',
    checkpoint_path: '/checkpoints/model-2.pt',
  },
  {
    id: 'model-3',
    name: 'Modulated SIREN Salt',
    type: 'modulated',
    architecture: { hidden_dim: 256, num_layers: 6, total_params: 786432 },
    training: { epochs: 1500, final_loss: 0.00312, physics_loss: 0.00201, data_loss: 0.00111 },
    velocity_model: 'salt_dome',
    created_at: '2024-01-13T09:15:00Z',
    checkpoint_path: '/checkpoints/model-3.pt',
  },
  {
    id: 'model-4',
    name: 'SIREN Elastic v2',
    type: 'siren',
    architecture: { hidden_dim: 384, num_layers: 8, total_params: 1179648 },
    training: { epochs: 3000, final_loss: 0.00156, physics_loss: 0.00089, data_loss: 0.00067 },
    velocity_model: 'marmousi',
    created_at: '2024-01-12T16:45:00Z',
    checkpoint_path: '/checkpoints/model-4.pt',
  },
];

function ModelCard({
  model,
  onSelect,
  onDelete,
}: {
  model: Model;
  onSelect: () => void;
  onDelete: () => void;
}) {
  const [showMenu, setShowMenu] = useState(false);

  const typeColors = {
    siren: 'from-primary-500 to-primary-600',
    fourier: 'from-accent-500 to-accent-600',
    modulated: 'from-green-500 to-green-600',
  };

  const formatParams = (params: number) => {
    if (params >= 1000000) return `${(params / 1000000).toFixed(1)}M`;
    if (params >= 1000) return `${(params / 1000).toFixed(0)}K`;
    return params.toString();
  };

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
      className="glass-panel p-6 hover:border-white/20 transition-all duration-300 group relative"
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div
            className={`w-12 h-12 rounded-xl bg-gradient-to-br ${typeColors[model.type]} flex items-center justify-center`}
          >
            <Box className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="font-semibold">{model.name}</h3>
            <p className="text-sm text-white/50">{model.type.toUpperCase()}</p>
          </div>
        </div>

        <div className="relative">
          <button
            onClick={() => setShowMenu(!showMenu)}
            className="p-2 hover:bg-white/10 rounded-lg transition-all opacity-0 group-hover:opacity-100"
          >
            <MoreVertical className="w-4 h-4" />
          </button>

          <AnimatePresence>
            {showMenu && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="absolute right-0 top-10 bg-surface-secondary border border-white/10 rounded-xl shadow-xl z-10 overflow-hidden min-w-[160px]"
              >
                <button
                  onClick={() => {
                    onSelect();
                    setShowMenu(false);
                  }}
                  className="w-full px-4 py-2 text-left hover:bg-white/10 transition-all flex items-center gap-2"
                >
                  <Info className="w-4 h-4" />
                  Details
                </button>
                <button className="w-full px-4 py-2 text-left hover:bg-white/10 transition-all flex items-center gap-2">
                  <Play className="w-4 h-4" />
                  Run Inference
                </button>
                <button className="w-full px-4 py-2 text-left hover:bg-white/10 transition-all flex items-center gap-2">
                  <Download className="w-4 h-4" />
                  Export
                </button>
                <button
                  onClick={() => {
                    onDelete();
                    setShowMenu(false);
                  }}
                  className="w-full px-4 py-2 text-left hover:bg-red-500/20 text-red-400 transition-all flex items-center gap-2"
                >
                  <Trash2 className="w-4 h-4" />
                  Delete
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Architecture Info */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="p-2 bg-white/5 rounded-lg">
          <p className="text-xs text-white/40">Hidden Dim</p>
          <p className="font-medium">{model.architecture.hidden_dim}</p>
        </div>
        <div className="p-2 bg-white/5 rounded-lg">
          <p className="text-xs text-white/40">Layers</p>
          <p className="font-medium">{model.architecture.num_layers}</p>
        </div>
        <div className="p-2 bg-white/5 rounded-lg">
          <p className="text-xs text-white/40">Params</p>
          <p className="font-medium">{formatParams(model.architecture.total_params)}</p>
        </div>
      </div>

      {/* Training Metrics */}
      <div className="space-y-2 mb-4">
        <div className="flex justify-between text-sm">
          <span className="text-white/60 flex items-center gap-1">
            <TrendingDown className="w-3 h-3" />
            Final Loss
          </span>
          <span className="font-mono">{model.training.final_loss.toExponential(3)}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-white/60">Epochs</span>
          <span>{model.training.epochs.toLocaleString()}</span>
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between pt-4 border-t border-white/10">
        <span className="text-xs text-white/40 flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {formatDate(model.created_at)}
        </span>
        <span className="text-xs px-2 py-1 bg-white/10 rounded-full">{model.velocity_model}</span>
      </div>
    </motion.div>
  );
}

function ModelDetails({ model, onClose }: { model: Model; onClose: () => void }) {
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
        className="glass-panel w-full max-w-2xl max-h-[80vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-6 border-b border-white/10 flex items-center justify-between sticky top-0 bg-surface-secondary/90 backdrop-blur-xl">
          <h2 className="text-xl font-semibold">{model.name}</h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/10 rounded-lg transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Model Type */}
          <div>
            <h3 className="text-sm text-white/60 mb-2">Model Type</h3>
            <span className="px-3 py-1 bg-primary-500/20 text-primary-400 rounded-full text-sm">
              {model.type.toUpperCase()}
            </span>
          </div>

          {/* Architecture */}
          <div>
            <h3 className="text-sm text-white/60 mb-3 flex items-center gap-2">
              <Layers className="w-4 h-4" />
              Architecture
            </h3>
            <div className="grid grid-cols-3 gap-4">
              <div className="p-4 bg-white/5 rounded-xl">
                <p className="text-xs text-white/40">Hidden Dimension</p>
                <p className="text-2xl font-bold">{model.architecture.hidden_dim}</p>
              </div>
              <div className="p-4 bg-white/5 rounded-xl">
                <p className="text-xs text-white/40">Number of Layers</p>
                <p className="text-2xl font-bold">{model.architecture.num_layers}</p>
              </div>
              <div className="p-4 bg-white/5 rounded-xl">
                <p className="text-xs text-white/40">Total Parameters</p>
                <p className="text-2xl font-bold">
                  {(model.architecture.total_params / 1e6).toFixed(2)}M
                </p>
              </div>
            </div>
          </div>

          {/* Training Metrics */}
          <div>
            <h3 className="text-sm text-white/60 mb-3 flex items-center gap-2">
              <TrendingDown className="w-4 h-4" />
              Training Metrics
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center p-3 bg-white/5 rounded-xl">
                <span>Final Loss</span>
                <span className="font-mono text-primary-400">
                  {model.training.final_loss.toExponential(4)}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-white/5 rounded-xl">
                <span>Physics Loss</span>
                <span className="font-mono text-accent-400">
                  {model.training.physics_loss.toExponential(4)}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-white/5 rounded-xl">
                <span>Data Loss</span>
                <span className="font-mono text-green-400">
                  {model.training.data_loss.toExponential(4)}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-white/5 rounded-xl">
                <span>Training Epochs</span>
                <span className="font-mono">{model.training.epochs.toLocaleString()}</span>
              </div>
            </div>
          </div>

          {/* Additional Info */}
          <div>
            <h3 className="text-sm text-white/60 mb-3">Additional Information</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-white/60">Velocity Model</span>
                <span>{model.velocity_model}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-white/60">Checkpoint Path</span>
                <span className="font-mono text-xs">{model.checkpoint_path}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-white/60">Created At</span>
                <span>{new Date(model.created_at).toLocaleString()}</span>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3 pt-4 border-t border-white/10">
            <button className="btn-primary flex-1 flex items-center justify-center gap-2">
              <Play className="w-4 h-4" />
              Run Inference
            </button>
            <button className="btn-secondary flex items-center gap-2">
              <Download className="w-4 h-4" />
              Export
            </button>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}

export default function Models() {
  const { data: apiModels } = useModels();
  const deleteMutation = useDeleteModel();

  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<string>('all');
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);

  // Use mock data if no API data
  const models = apiModels?.length ? apiModels : mockModels;

  const filteredModels = models.filter((model) => {
    const matchesSearch = model.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = filterType === 'all' || model.type === filterType;
    return matchesSearch && matchesType;
  });

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
          <h1 className="text-3xl font-bold">Models</h1>
          <p className="text-white/60 mt-1">
            Manage your trained PINN models
          </p>
        </div>

        <div className="text-right">
          <p className="text-2xl font-bold">{models.length}</p>
          <p className="text-sm text-white/60">Saved Models</p>
        </div>
      </motion.div>

      {/* Filters */}
      <motion.div variants={item} className="glass-panel p-4 flex items-center gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-white/40" />
          <input
            type="text"
            placeholder="Search models..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="input-field pl-10"
          />
        </div>

        <div className="flex items-center gap-2">
          <Filter className="w-5 h-5 text-white/40" />
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="bg-white/10 border border-white/20 rounded-xl px-4 py-2"
          >
            <option value="all">All Types</option>
            <option value="siren">SIREN</option>
            <option value="fourier">Fourier</option>
            <option value="modulated">Modulated</option>
          </select>
        </div>
      </motion.div>

      {/* Models Grid */}
      <motion.div
        variants={container}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
      >
        {filteredModels.length === 0 ? (
          <motion.div
            variants={item}
            className="col-span-full glass-panel p-12 text-center"
          >
            <Box className="w-16 h-16 text-white/20 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">No models found</h3>
            <p className="text-white/60">
              {searchQuery || filterType !== 'all'
                ? 'Try adjusting your search or filters'
                : 'Train your first model to get started'}
            </p>
          </motion.div>
        ) : (
          filteredModels.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              onSelect={() => setSelectedModel(model)}
              onDelete={() => deleteMutation.mutate(model.id)}
            />
          ))
        )}
      </motion.div>

      {/* Model Details Modal */}
      <AnimatePresence>
        {selectedModel && (
          <ModelDetails
            model={selectedModel}
            onClose={() => setSelectedModel(null)}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
}
