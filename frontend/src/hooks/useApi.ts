import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { TrainingConfig, TrainingJob, Model, Experiment, SystemStatus, VelocityModel } from '../types';
import config from '../config';

const API_BASE = config.apiBase;

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

// System Status
export function useSystemStatus() {
  return useQuery({
    queryKey: ['systemStatus'],
    queryFn: () => fetchJson<SystemStatus>('/status'),
    refetchInterval: 5000,
  });
}

// Training Jobs
export function useTrainingJobs() {
  return useQuery({
    queryKey: ['trainingJobs'],
    queryFn: () => fetchJson<TrainingJob[]>('/training/jobs'),
    refetchInterval: 3000,
  });
}

export function useTrainingJob(jobId: string | null) {
  return useQuery({
    queryKey: ['trainingJob', jobId],
    queryFn: () => fetchJson<TrainingJob>(`/training/jobs/${jobId}`),
    enabled: !!jobId,
    refetchInterval: 2000,
  });
}

export function useStartTraining() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (config: TrainingConfig) =>
      fetchJson<{ job_id: string }>('/training/start', {
        method: 'POST',
        body: JSON.stringify(config),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trainingJobs'] });
    },
  });
}

export function usePauseTraining() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) =>
      fetchJson<void>(`/training/jobs/${jobId}/pause`, { method: 'POST' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trainingJobs'] });
    },
  });
}

export function useResumeTraining() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) =>
      fetchJson<void>(`/training/jobs/${jobId}/resume`, { method: 'POST' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trainingJobs'] });
    },
  });
}

export function useCancelTraining() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) =>
      fetchJson<void>(`/training/jobs/${jobId}/cancel`, { method: 'POST' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trainingJobs'] });
    },
  });
}

// Models
export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: () => fetchJson<Model[]>('/models'),
  });
}

export function useModel(modelId: string | null) {
  return useQuery({
    queryKey: ['model', modelId],
    queryFn: () => fetchJson<Model>(`/models/${modelId}`),
    enabled: !!modelId,
  });
}

export function useDeleteModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (modelId: string) =>
      fetchJson<void>(`/models/${modelId}`, { method: 'DELETE' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });
}

// Experiments
export function useExperiments() {
  return useQuery({
    queryKey: ['experiments'],
    queryFn: () => fetchJson<Experiment[]>('/experiments'),
  });
}

export function useExperiment(experimentId: string | null) {
  return useQuery({
    queryKey: ['experiment', experimentId],
    queryFn: () => fetchJson<Experiment>(`/experiments/${experimentId}`),
    enabled: !!experimentId,
  });
}

export function useCreateExperiment() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: { name: string; description: string; model_ids: string[] }) =>
      fetchJson<Experiment>('/experiments', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['experiments'] });
    },
  });
}

// Velocity Models
export function useVelocityModels() {
  return useQuery({
    queryKey: ['velocityModels'],
    queryFn: () => fetchJson<VelocityModel[]>('/velocity-models'),
  });
}

export function useGenerateVelocityModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: { type: string; params: Record<string, unknown> }) =>
      fetchJson<VelocityModel>('/velocity-models/generate', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['velocityModels'] });
    },
  });
}

// Inference
export function useInference() {
  return useMutation({
    mutationFn: (data: { model_id: string; x: number[]; z: number[]; t: number[] }) =>
      fetchJson<{ wavefield: number[][] }>('/inference', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
  });
}

// Architecture Info
export function useArchitectures() {
  return useQuery({
    queryKey: ['architectures'],
    queryFn: () =>
      fetchJson<{
        models: string[];
        wave_equations: string[];
        sources: string[];
        boundaries: string[];
        adaptive_weights: string[];
        schedulers: string[];
      }>('/architectures'),
  });
}
