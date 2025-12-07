// API and Model Types

export interface TrainingConfig {
  model_type: 'siren' | 'fourier' | 'modulated';
  hidden_dim: number;
  num_layers: number;
  learning_rate: number;
  batch_size: number;
  num_epochs: number;
  physics_weight: number;
  data_weight: number;
  boundary_weight: number;
  use_adaptive_weights: boolean;
  adaptive_method: 'gradnorm' | 'uncertainty' | 'softadapt';
  scheduler: 'cosine' | 'cyclic' | 'step' | 'plateau';
  velocity_model: 'marmousi' | 'layered' | 'random' | 'salt_dome' | 'fault';
  wave_equation: 'acoustic' | 'elastic' | 'viscoacoustic';
  domain_x: [number, number];
  domain_z: [number, number];
  domain_t: [number, number];
  source_frequency: number;
  source_position: [number, number];
}

export interface TrainingProgress {
  epoch: number;
  total_epochs: number;
  step: number;
  total_steps: number;
  loss: number;
  physics_loss: number;
  data_loss: number;
  boundary_loss: number;
  learning_rate: number;
  elapsed_time: number;
  eta: number;
  metrics: Record<string, number>;
}

export interface TrainingJob {
  job_id: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  config: TrainingConfig;
  progress: TrainingProgress | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
}

export interface Model {
  id: string;
  name: string;
  type: 'siren' | 'fourier' | 'modulated';
  architecture: {
    hidden_dim: number;
    num_layers: number;
    total_params: number;
  };
  training: {
    epochs: number;
    final_loss: number;
    physics_loss: number;
    data_loss: number;
  };
  velocity_model: string;
  created_at: string;
  checkpoint_path: string;
}

export interface Experiment {
  id: string;
  name: string;
  description: string;
  status: 'running' | 'completed' | 'failed';
  models: string[];
  metrics: Record<string, number>;
  created_at: string;
  updated_at: string;
}

export interface VelocityModel {
  name: string;
  type: string;
  shape: [number, number];
  min_velocity: number;
  max_velocity: number;
  data: number[][] | null;
}

export interface WavefieldSnapshot {
  time: number;
  data: number[][];
  x_coords: number[];
  z_coords: number[];
}

export interface Seismogram {
  receiver_positions: number[];
  time_samples: number[];
  data: number[][];
}

export interface SystemStatus {
  api_connected: boolean;
  gpu_available: boolean;
  gpu_name: string | null;
  gpu_memory_total: number | null;
  gpu_memory_used: number | null;
  active_jobs: number;
  completed_jobs: number;
}

export interface MetricHistory {
  epoch: number;
  loss: number;
  physics_loss: number;
  data_loss: number;
  boundary_loss: number;
  learning_rate: number;
}

export interface ChartData {
  name: string;
  value: number;
  [key: string]: string | number;
}
