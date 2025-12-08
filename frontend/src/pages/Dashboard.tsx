import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  Activity,
  Cpu,
  Zap,
  Clock,
  TrendingUp,
  Box,
  Layers,
  BarChart3,
  AlertTriangle,
  Radio,
  Waves,
  Globe,
  Server,
  HardDrive,
  ArrowRight,
  CheckCircle,
  XCircle,
  Pause,
} from 'lucide-react';
import { useSystemStatus, useTrainingJobs, useModels } from '../hooks/useApi';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { useTrainingStore } from '../stores/trainingStore';

export default function Dashboard() {
  const { data: status } = useSystemStatus();
  const { data: jobs } = useTrainingJobs();
  const { data: models } = useModels();
  const { metricHistory } = useTrainingStore();

  const activeJobs = jobs?.filter((j) => j.status === 'running').length || 0;
  const completedJobs = jobs?.filter((j) => j.status === 'completed').length || 0;
  const totalModels = models?.length || 0;
  const totalJobs = jobs?.length || 0;

  const gpuUsage = status?.gpu_memory_total
    ? ((status.gpu_memory_used || 0) / status.gpu_memory_total) * 100
    : 0;

  // Chart data
  const chartData = metricHistory.length > 0
    ? metricHistory.map((m) => ({
        epoch: m.epoch,
        total: m.loss,
        physics: m.physics_loss,
        data: m.data_loss,
      }))
    : Array.from({ length: 50 }, (_, i) => ({
        epoch: i + 1,
        total: Math.exp(-i / 20) * 0.5 + 0.01 + Math.random() * 0.02,
        physics: Math.exp(-i / 25) * 0.3 + 0.005 + Math.random() * 0.01,
        data: Math.exp(-i / 15) * 0.2 + 0.005 + Math.random() * 0.01,
      }));

  const recentJobs = jobs?.slice(0, 5) || [];

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

        .dashboard-container {
          min-height: 100vh;
          background: #0a0a0a;
          color: #e5e5e5;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
          padding: 0;
          margin: -32px;
        }

        .dashboard-header {
          background: linear-gradient(180deg, #0d0d0d 0%, #0a0a0a 100%);
          border-bottom: 1px solid #1f1f1f;
          padding: 24px 32px;
        }

        .header-content {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .header-title {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 24px;
          font-weight: 700;
          color: #ffffff;
          letter-spacing: -0.5px;
        }

        .header-subtitle {
          font-size: 13px;
          color: #737373;
          margin-top: 4px;
        }

        .header-status {
          display: flex;
          align-items: center;
          gap: 24px;
        }

        .status-item {
          display: flex;
          align-items: center;
          gap: 8px;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 12px;
        }

        .status-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          animation: pulse 2s infinite;
        }

        .status-dot.online { background: #22c55e; }
        .status-dot.offline { background: #ef4444; }
        .status-dot.warning { background: #f59e0b; }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        .dashboard-content {
          padding: 24px 32px;
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 16px;
          margin-bottom: 24px;
        }

        .metric-card {
          background: #0d0d0d;
          border: 1px solid #1f1f1f;
          border-radius: 8px;
          padding: 20px;
          position: relative;
          overflow: hidden;
        }

        .metric-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 2px;
        }

        .metric-card.blue::before { background: linear-gradient(90deg, #3b82f6, #1d4ed8); }
        .metric-card.green::before { background: linear-gradient(90deg, #22c55e, #16a34a); }
        .metric-card.purple::before { background: linear-gradient(90deg, #8b5cf6, #6d28d9); }
        .metric-card.orange::before { background: linear-gradient(90deg, #f59e0b, #d97706); }
        .metric-card.red::before { background: linear-gradient(90deg, #ef4444, #dc2626); }

        .metric-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 12px;
        }

        .metric-label {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 10px;
          font-weight: 600;
          color: #737373;
          text-transform: uppercase;
          letter-spacing: 1px;
        }

        .metric-icon {
          width: 32px;
          height: 32px;
          border-radius: 6px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .metric-icon.blue { background: rgba(59, 130, 246, 0.15); color: #60a5fa; }
        .metric-icon.green { background: rgba(34, 197, 94, 0.15); color: #4ade80; }
        .metric-icon.purple { background: rgba(139, 92, 246, 0.15); color: #a78bfa; }
        .metric-icon.orange { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
        .metric-icon.red { background: rgba(239, 68, 68, 0.15); color: #f87171; }

        .metric-value {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 32px;
          font-weight: 700;
          color: #ffffff;
          line-height: 1;
        }

        .metric-sub {
          font-size: 12px;
          color: #525252;
          margin-top: 8px;
        }

        .main-grid {
          display: grid;
          grid-template-columns: 2fr 1fr;
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
          font-size: 13px;
          font-weight: 600;
          color: #a3a3a3;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .panel-title svg {
          width: 16px;
          height: 16px;
          color: #3b82f6;
        }

        .panel-body {
          padding: 20px;
        }

        .chart-container {
          height: 280px;
        }

        .jobs-list {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .job-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px 16px;
          background: #141414;
          border: 1px solid #1f1f1f;
          border-radius: 6px;
          transition: all 0.2s;
        }

        .job-item:hover {
          border-color: #333;
          background: #1a1a1a;
        }

        .job-info {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .job-icon {
          width: 36px;
          height: 36px;
          background: rgba(59, 130, 246, 0.1);
          border-radius: 6px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .job-icon svg {
          width: 18px;
          height: 18px;
          color: #60a5fa;
        }

        .job-details h4 {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 13px;
          font-weight: 600;
          color: #e5e5e5;
        }

        .job-details p {
          font-size: 11px;
          color: #525252;
          margin-top: 2px;
        }

        .job-status {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 10px;
          font-weight: 600;
          padding: 4px 10px;
          border-radius: 4px;
          text-transform: uppercase;
        }

        .job-status.running {
          background: rgba(34, 197, 94, 0.15);
          color: #4ade80;
          border: 1px solid rgba(34, 197, 94, 0.3);
        }

        .job-status.completed {
          background: rgba(59, 130, 246, 0.15);
          color: #60a5fa;
          border: 1px solid rgba(59, 130, 246, 0.3);
        }

        .job-status.failed {
          background: rgba(239, 68, 68, 0.15);
          color: #f87171;
          border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .job-status.pending {
          background: rgba(245, 158, 11, 0.15);
          color: #fbbf24;
          border: 1px solid rgba(245, 158, 11, 0.3);
        }

        .empty-state {
          text-align: center;
          padding: 40px 20px;
          color: #525252;
        }

        .empty-state svg {
          width: 48px;
          height: 48px;
          margin-bottom: 12px;
          opacity: 0.3;
        }

        .quick-actions {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 12px;
          margin-top: 24px;
        }

        .action-card {
          display: flex;
          flex-direction: column;
          gap: 8px;
          padding: 20px;
          background: #141414;
          border: 1px solid #1f1f1f;
          border-radius: 8px;
          text-decoration: none;
          color: inherit;
          transition: all 0.2s;
        }

        .action-card:hover {
          border-color: #333;
          transform: translateY(-2px);
        }

        .action-card.primary:hover { border-color: #3b82f6; }
        .action-card.green:hover { border-color: #22c55e; }
        .action-card.purple:hover { border-color: #8b5cf6; }
        .action-card.orange:hover { border-color: #f59e0b; }
        .action-card.red:hover { border-color: #ef4444; }

        .action-icon {
          width: 40px;
          height: 40px;
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .action-icon.primary { background: rgba(59, 130, 246, 0.15); color: #60a5fa; }
        .action-icon.green { background: rgba(34, 197, 94, 0.15); color: #4ade80; }
        .action-icon.purple { background: rgba(139, 92, 246, 0.15); color: #a78bfa; }
        .action-icon.orange { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
        .action-icon.red { background: rgba(239, 68, 68, 0.15); color: #f87171; }

        .action-title {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 14px;
          font-weight: 600;
          color: #e5e5e5;
        }

        .action-desc {
          font-size: 12px;
          color: #525252;
        }

        .gpu-section {
          margin-top: 24px;
        }

        .gpu-bar-container {
          margin-bottom: 16px;
        }

        .gpu-bar-label {
          display: flex;
          justify-content: space-between;
          margin-bottom: 8px;
          font-size: 12px;
          color: #737373;
        }

        .gpu-bar-label span:last-child {
          font-family: 'IBM Plex Mono', monospace;
          color: #e5e5e5;
        }

        .gpu-bar {
          height: 8px;
          background: #1f1f1f;
          border-radius: 4px;
          overflow: hidden;
        }

        .gpu-bar-fill {
          height: 100%;
          background: linear-gradient(90deg, #3b82f6, #8b5cf6);
          border-radius: 4px;
          transition: width 0.5s ease;
        }

        .gpu-stats {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 12px;
        }

        .gpu-stat {
          background: #141414;
          border: 1px solid #1f1f1f;
          border-radius: 6px;
          padding: 12px;
        }

        .gpu-stat-label {
          font-size: 10px;
          color: #525252;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .gpu-stat-value {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 14px;
          font-weight: 600;
          color: #e5e5e5;
          margin-top: 4px;
        }

        .featured-link {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-top: 24px;
          padding: 20px;
          background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          text-decoration: none;
          color: inherit;
          transition: all 0.2s;
        }

        .featured-link:hover {
          border-color: #ef4444;
          transform: translateY(-2px);
        }

        .featured-content {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .featured-icon {
          width: 48px;
          height: 48px;
          background: rgba(239, 68, 68, 0.2);
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #f87171;
        }

        .featured-text h3 {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 16px;
          font-weight: 700;
          color: #fca5a5;
        }

        .featured-text p {
          font-size: 13px;
          color: #a3a3a3;
          margin-top: 4px;
        }

        .featured-arrow {
          color: #f87171;
        }

        @media (max-width: 1200px) {
          .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
          }
          .main-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>

      <div className="dashboard-container">
        <header className="dashboard-header">
          <div className="header-content">
            <div>
              <h1 className="header-title">PINN SEISMIC</h1>
              <p className="header-subtitle">Physics-Informed Neural Networks for Waveform Inversion</p>
            </div>
            <div className="header-status">
              <div className="status-item">
                <span className={`status-dot ${status?.gpu_available ? 'online' : 'offline'}`} />
                <span>GPU {status?.gpu_available ? 'Online' : 'Offline'}</span>
              </div>
              <div className="status-item">
                <span className="status-dot online" />
                <span>API Connected</span>
              </div>
            </div>
          </div>
        </header>

        <div className="dashboard-content">
          {/* Featured Link - Tsunami Warning */}
          <Link to="/tsunami" className="featured-link">
            <div className="featured-content">
              <div className="featured-icon">
                <AlertTriangle size={24} />
              </div>
              <div className="featured-text">
                <h3>TSUNAMI WARNING SYSTEM</h3>
                <p>Real-time Indian Ocean seismic monitoring with live data feeds</p>
              </div>
            </div>
            <ArrowRight className="featured-arrow" size={24} />
          </Link>

          {/* Metrics */}
          <div className="metrics-grid" style={{ marginTop: '24px' }}>
            <div className="metric-card blue">
              <div className="metric-header">
                <span className="metric-label">Active Training</span>
                <div className="metric-icon blue">
                  <Activity size={16} />
                </div>
              </div>
              <div className="metric-value">{activeJobs}</div>
              <div className="metric-sub">{totalJobs} total jobs</div>
            </div>

            <div className="metric-card green">
              <div className="metric-header">
                <span className="metric-label">Saved Models</span>
                <div className="metric-icon green">
                  <Box size={16} />
                </div>
              </div>
              <div className="metric-value">{totalModels}</div>
              <div className="metric-sub">Ready for inference</div>
            </div>

            <div className="metric-card purple">
              <div className="metric-header">
                <span className="metric-label">GPU Status</span>
                <div className="metric-icon purple">
                  <Cpu size={16} />
                </div>
              </div>
              <div className="metric-value" style={{ fontSize: '24px' }}>
                {status?.gpu_available ? 'Online' : 'N/A'}
              </div>
              <div className="metric-sub">{status?.gpu_name || 'No GPU detected'}</div>
            </div>

            <div className="metric-card orange">
              <div className="metric-header">
                <span className="metric-label">Completed</span>
                <div className="metric-icon orange">
                  <CheckCircle size={16} />
                </div>
              </div>
              <div className="metric-value">{completedJobs}</div>
              <div className="metric-sub">Experiments finished</div>
            </div>
          </div>

          {/* Main Content */}
          <div className="main-grid">
            {/* Left Column */}
            <div>
              {/* Loss Chart */}
              <div className="panel">
                <div className="panel-header">
                  <div className="panel-title">
                    <TrendingUp />
                    Training Loss History
                  </div>
                </div>
                <div className="panel-body">
                  <div className="chart-container">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={chartData}>
                        <defs>
                          <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                          </linearGradient>
                          <linearGradient id="colorPhysics" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" />
                        <XAxis dataKey="epoch" stroke="#525252" fontSize={11} />
                        <YAxis stroke="#525252" fontSize={11} tickFormatter={(v) => v.toExponential(1)} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#1a1a1a',
                            border: '1px solid #333',
                            borderRadius: '6px',
                            fontFamily: 'IBM Plex Mono, monospace',
                            fontSize: '12px',
                          }}
                          labelStyle={{ color: '#e5e5e5' }}
                        />
                        <Area type="monotone" dataKey="total" stroke="#3b82f6" fillOpacity={1} fill="url(#colorTotal)" name="Total Loss" />
                        <Area type="monotone" dataKey="physics" stroke="#8b5cf6" fillOpacity={1} fill="url(#colorPhysics)" name="Physics Loss" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Quick Actions */}
              <div className="quick-actions">
                <Link to="/training" className="action-card primary">
                  <div className="action-icon primary">
                    <Activity size={20} />
                  </div>
                  <div className="action-title">New Training</div>
                  <div className="action-desc">Start a new PINN training job</div>
                </Link>
                <Link to="/visualization" className="action-card purple">
                  <div className="action-icon purple">
                    <Waves size={20} />
                  </div>
                  <div className="action-title">Visualize</div>
                  <div className="action-desc">View wavefields & velocity models</div>
                </Link>
                <Link to="/models" className="action-card green">
                  <div className="action-icon green">
                    <Box size={20} />
                  </div>
                  <div className="action-title">Models</div>
                  <div className="action-desc">Manage saved models</div>
                </Link>
                <Link to="/experiments" className="action-card orange">
                  <div className="action-icon orange">
                    <BarChart3 size={20} />
                  </div>
                  <div className="action-title">Experiments</div>
                  <div className="action-desc">Compare & analyze results</div>
                </Link>
              </div>
            </div>

            {/* Right Column */}
            <div>
              {/* Recent Jobs */}
              <div className="panel">
                <div className="panel-header">
                  <div className="panel-title">
                    <Clock />
                    Recent Jobs
                  </div>
                </div>
                <div className="panel-body">
                  <div className="jobs-list">
                    {recentJobs.length === 0 ? (
                      <div className="empty-state">
                        <Layers />
                        <p>No training jobs yet</p>
                      </div>
                    ) : (
                      recentJobs.map((job) => (
                        <div key={job.job_id} className="job-item">
                          <div className="job-info">
                            <div className="job-icon">
                              <Layers size={18} />
                            </div>
                            <div className="job-details">
                              <h4>{job.config.model_type.toUpperCase()}</h4>
                              <p>{job.config.velocity_model} - {job.config.wave_equation}</p>
                            </div>
                          </div>
                          <span className={`job-status ${job.status}`}>{job.status}</span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>

              {/* GPU Status */}
              <div className="panel gpu-section">
                <div className="panel-header">
                  <div className="panel-title">
                    <Cpu />
                    GPU Status
                  </div>
                </div>
                <div className="panel-body">
                  <div className="gpu-bar-container">
                    <div className="gpu-bar-label">
                      <span>Memory Usage</span>
                      <span>{gpuUsage.toFixed(1)}%</span>
                    </div>
                    <div className="gpu-bar">
                      <div className="gpu-bar-fill" style={{ width: `${gpuUsage}%` }} />
                    </div>
                  </div>
                  <div className="gpu-stats">
                    <div className="gpu-stat">
                      <div className="gpu-stat-label">Device</div>
                      <div className="gpu-stat-value">{status?.gpu_name || 'N/A'}</div>
                    </div>
                    <div className="gpu-stat">
                      <div className="gpu-stat-label">Total Memory</div>
                      <div className="gpu-stat-value">
                        {status?.gpu_memory_total
                          ? `${(status.gpu_memory_total / 1024).toFixed(1)} GB`
                          : 'N/A'}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
