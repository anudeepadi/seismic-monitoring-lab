import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { TrainingJob, TrainingProgress, MetricHistory, TrainingConfig } from '../types';

interface TrainingState {
  // Current training state
  activeJob: TrainingJob | null;
  progress: TrainingProgress | null;
  metricHistory: MetricHistory[];

  // Job management
  jobs: TrainingJob[];
  isLoading: boolean;
  error: string | null;

  // WebSocket connection
  wsConnected: boolean;

  // Actions
  setActiveJob: (job: TrainingJob | null) => void;
  updateProgress: (progress: TrainingProgress) => void;
  addMetricHistory: (metric: MetricHistory) => void;
  clearMetricHistory: () => void;
  setJobs: (jobs: TrainingJob[]) => void;
  addJob: (job: TrainingJob) => void;
  updateJob: (jobId: string, updates: Partial<TrainingJob>) => void;
  removeJob: (jobId: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setWsConnected: (connected: boolean) => void;
  reset: () => void;
}

const initialState = {
  activeJob: null,
  progress: null,
  metricHistory: [],
  jobs: [],
  isLoading: false,
  error: null,
  wsConnected: false,
};

export const useTrainingStore = create<TrainingState>()(
  subscribeWithSelector((set, get) => ({
    ...initialState,

    setActiveJob: (job) => set({ activeJob: job }),

    updateProgress: (progress) => {
      set({ progress });
      // Also add to metric history
      const metric: MetricHistory = {
        epoch: progress.epoch,
        loss: progress.loss,
        physics_loss: progress.physics_loss,
        data_loss: progress.data_loss,
        boundary_loss: progress.boundary_loss,
        learning_rate: progress.learning_rate,
      };
      get().addMetricHistory(metric);
    },

    addMetricHistory: (metric) =>
      set((state) => {
        // Avoid duplicates for same epoch
        const exists = state.metricHistory.some((m) => m.epoch === metric.epoch);
        if (exists) return state;
        return { metricHistory: [...state.metricHistory, metric] };
      }),

    clearMetricHistory: () => set({ metricHistory: [] }),

    setJobs: (jobs) => set({ jobs }),

    addJob: (job) =>
      set((state) => ({
        jobs: [job, ...state.jobs],
      })),

    updateJob: (jobId, updates) =>
      set((state) => ({
        jobs: state.jobs.map((job) =>
          job.job_id === jobId ? { ...job, ...updates } : job
        ),
      })),

    removeJob: (jobId) =>
      set((state) => ({
        jobs: state.jobs.filter((job) => job.job_id !== jobId),
      })),

    setLoading: (isLoading) => set({ isLoading }),

    setError: (error) => set({ error }),

    setWsConnected: (wsConnected) => set({ wsConnected }),

    reset: () => set(initialState),
  }))
);

// Default training configuration
export const defaultTrainingConfig: TrainingConfig = {
  model_type: 'siren',
  hidden_dim: 256,
  num_layers: 6,
  learning_rate: 1e-4,
  batch_size: 4096,
  num_epochs: 1000,
  physics_weight: 1.0,
  data_weight: 1.0,
  boundary_weight: 0.1,
  use_adaptive_weights: true,
  adaptive_method: 'gradnorm',
  scheduler: 'cosine',
  velocity_model: 'marmousi',
  wave_equation: 'acoustic',
  domain_x: [0, 2000],
  domain_z: [0, 1000],
  domain_t: [0, 1.0],
  source_frequency: 15.0,
  source_position: [1000, 50],
};
