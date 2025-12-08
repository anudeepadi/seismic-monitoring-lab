// API Configuration
// Uses environment variables in production, falls back to localhost for development

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

export const config = {
  apiBase: `${API_URL}/api`,
  apiBaseV1: `${API_URL}/api/v1`,
  wsBase: WS_URL,
  wsLive: `${WS_URL}/api/v1/streams/live`,
};

export default config;
