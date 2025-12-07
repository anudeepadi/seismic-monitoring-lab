import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Model, Experiment, SystemStatus, VelocityModel } from '../types';

interface AppState {
  // System status
  systemStatus: SystemStatus;

  // Models
  models: Model[];
  selectedModel: Model | null;

  // Experiments
  experiments: Experiment[];
  selectedExperiment: Experiment | null;

  // Velocity models
  velocityModels: VelocityModel[];
  selectedVelocityModel: VelocityModel | null;

  // UI State
  sidebarCollapsed: boolean;
  theme: 'dark' | 'light';

  // Actions
  setSystemStatus: (status: SystemStatus) => void;
  setModels: (models: Model[]) => void;
  addModel: (model: Model) => void;
  setSelectedModel: (model: Model | null) => void;
  setExperiments: (experiments: Experiment[]) => void;
  addExperiment: (experiment: Experiment) => void;
  setSelectedExperiment: (experiment: Experiment | null) => void;
  setVelocityModels: (models: VelocityModel[]) => void;
  setSelectedVelocityModel: (model: VelocityModel | null) => void;
  toggleSidebar: () => void;
  setTheme: (theme: 'dark' | 'light') => void;
}

const defaultSystemStatus: SystemStatus = {
  api_connected: false,
  gpu_available: false,
  gpu_name: null,
  gpu_memory_total: null,
  gpu_memory_used: null,
  active_jobs: 0,
  completed_jobs: 0,
};

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Initial state
      systemStatus: defaultSystemStatus,
      models: [],
      selectedModel: null,
      experiments: [],
      selectedExperiment: null,
      velocityModels: [],
      selectedVelocityModel: null,
      sidebarCollapsed: false,
      theme: 'dark',

      // Actions
      setSystemStatus: (systemStatus) => set({ systemStatus }),

      setModels: (models) => set({ models }),

      addModel: (model) =>
        set((state) => ({
          models: [model, ...state.models],
        })),

      setSelectedModel: (selectedModel) => set({ selectedModel }),

      setExperiments: (experiments) => set({ experiments }),

      addExperiment: (experiment) =>
        set((state) => ({
          experiments: [experiment, ...state.experiments],
        })),

      setSelectedExperiment: (selectedExperiment) => set({ selectedExperiment }),

      setVelocityModels: (velocityModels) => set({ velocityModels }),

      setSelectedVelocityModel: (selectedVelocityModel) =>
        set({ selectedVelocityModel }),

      toggleSidebar: () =>
        set((state) => ({
          sidebarCollapsed: !state.sidebarCollapsed,
        })),

      setTheme: (theme) => set({ theme }),
    }),
    {
      name: 'pinn-seismic-app-store',
      partialize: (state) => ({
        sidebarCollapsed: state.sidebarCollapsed,
        theme: state.theme,
      }),
    }
  )
);
