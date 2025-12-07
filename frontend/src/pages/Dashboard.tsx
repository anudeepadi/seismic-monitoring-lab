import { motion } from 'framer-motion';
import {
  Activity,
  Cpu,
  Zap,
  Clock,
  TrendingUp,
  Box,
  Layers,
  BarChart3,
} from 'lucide-react';
import { useSystemStatus, useTrainingJobs, useModels } from '../hooks/useApi';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { useTrainingStore } from '../stores/trainingStore';

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

function MetricCard({
  icon: Icon,
  label,
  value,
  subValue,
  color = 'primary',
}: {
  icon: React.ElementType;
  label: string;
  value: string | number;
  subValue?: string;
  color?: 'primary' | 'accent' | 'green' | 'yellow' | 'red';
}) {
  const colorClasses = {
    primary: 'from-primary-500 to-primary-600',
    accent: 'from-accent-500 to-accent-600',
    green: 'from-green-500 to-green-600',
    yellow: 'from-yellow-500 to-yellow-600',
    red: 'from-red-500 to-red-600',
  };

  return (
    <motion.div variants={item} className="glass-panel p-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-white/60 uppercase tracking-wider">{label}</p>
          <p className="text-3xl font-bold mt-2">{value}</p>
          {subValue && (
            <p className="text-sm text-white/40 mt-1">{subValue}</p>
          )}
        </div>
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${colorClasses[color]} flex items-center justify-center`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
    </motion.div>
  );
}

function RecentJobsList() {
  const { data: jobs } = useTrainingJobs();

  const recentJobs = jobs?.slice(0, 5) || [];

  const statusColors = {
    running: 'status-running',
    pending: 'status-pending',
    completed: 'status-completed',
    failed: 'status-failed',
    paused: 'status-pending',
    cancelled: 'status-failed',
  };

  return (
    <motion.div variants={item} className="glass-panel p-6">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Clock className="w-5 h-5 text-primary-400" />
        Recent Training Jobs
      </h3>
      <div className="space-y-3">
        {recentJobs.length === 0 ? (
          <p className="text-white/40 text-sm">No training jobs yet</p>
        ) : (
          recentJobs.map((job) => (
            <div
              key={job.job_id}
              className="flex items-center justify-between p-3 bg-white/5 rounded-xl"
            >
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary-500/20 to-accent-500/20 flex items-center justify-center">
                  <Layers className="w-5 h-5 text-primary-400" />
                </div>
                <div>
                  <p className="font-medium">{job.config.model_type.toUpperCase()}</p>
                  <p className="text-xs text-white/40">
                    {job.config.velocity_model} - {job.config.wave_equation}
                  </p>
                </div>
              </div>
              <span className={statusColors[job.status] || 'status-pending'}>
                {job.status}
              </span>
            </div>
          ))
        )}
      </div>
    </motion.div>
  );
}

function LossChart() {
  const { metricHistory } = useTrainingStore();

  // Use mock data if no history
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

  return (
    <motion.div variants={item} className="glass-panel p-6">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <TrendingUp className="w-5 h-5 text-primary-400" />
        Training Loss History
      </h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorPhysics" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#d946ef" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#d946ef" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis
              dataKey="epoch"
              stroke="rgba(255,255,255,0.4)"
              fontSize={12}
            />
            <YAxis
              stroke="rgba(255,255,255,0.4)"
              fontSize={12}
              tickFormatter={(v) => v.toExponential(1)}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '12px',
              }}
              labelStyle={{ color: 'white' }}
            />
            <Area
              type="monotone"
              dataKey="total"
              stroke="#0ea5e9"
              fillOpacity={1}
              fill="url(#colorTotal)"
              name="Total Loss"
            />
            <Area
              type="monotone"
              dataKey="physics"
              stroke="#d946ef"
              fillOpacity={1}
              fill="url(#colorPhysics)"
              name="Physics Loss"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}

function GPUMetrics() {
  const { data: status } = useSystemStatus();

  const gpuUsage = status?.gpu_memory_total
    ? ((status.gpu_memory_used || 0) / status.gpu_memory_total) * 100
    : 0;

  return (
    <motion.div variants={item} className="glass-panel p-6">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Cpu className="w-5 h-5 text-primary-400" />
        GPU Status
      </h3>
      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-white/60">Memory Usage</span>
            <span>{gpuUsage.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-primary-500 to-accent-500"
              initial={{ width: 0 }}
              animate={{ width: `${gpuUsage}%` }}
              transition={{ duration: 1, ease: 'easeOut' }}
            />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 bg-white/5 rounded-xl">
            <p className="text-xs text-white/40">Device</p>
            <p className="font-medium truncate">{status?.gpu_name || 'Not available'}</p>
          </div>
          <div className="p-3 bg-white/5 rounded-xl">
            <p className="text-xs text-white/40">Total Memory</p>
            <p className="font-medium">
              {status?.gpu_memory_total
                ? `${(status.gpu_memory_total / 1024).toFixed(1)} GB`
                : 'N/A'}
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export default function Dashboard() {
  const { data: status } = useSystemStatus();
  const { data: jobs } = useTrainingJobs();
  const { data: models } = useModels();

  const activeJobs = jobs?.filter((j) => j.status === 'running').length || 0;
  const completedJobs = jobs?.filter((j) => j.status === 'completed').length || 0;
  const totalModels = models?.length || 0;

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="space-y-8"
    >
      {/* Header */}
      <motion.div variants={item}>
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-white/60 mt-1">
          Physics-Informed Neural Networks for Seismic Inversion
        </p>
      </motion.div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          icon={Activity}
          label="Active Training"
          value={activeJobs}
          subValue={`${jobs?.length || 0} total jobs`}
          color="green"
        />
        <MetricCard
          icon={Box}
          label="Saved Models"
          value={totalModels}
          subValue="Ready for inference"
          color="primary"
        />
        <MetricCard
          icon={Zap}
          label="GPU Status"
          value={status?.gpu_available ? 'Available' : 'Unavailable'}
          subValue={status?.gpu_name || 'No GPU detected'}
          color={status?.gpu_available ? 'green' : 'yellow'}
        />
        <MetricCard
          icon={BarChart3}
          label="Completed"
          value={completedJobs}
          subValue="Experiments finished"
          color="accent"
        />
      </div>

      {/* Charts and Lists */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <LossChart />
        <GPUMetrics />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RecentJobsList />

        {/* Quick Actions */}
        <motion.div variants={item} className="glass-panel p-6">
          <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
          <div className="grid grid-cols-2 gap-4">
            <a
              href="/training"
              className="p-4 bg-gradient-to-br from-primary-500/20 to-primary-600/10 border border-primary-500/30 rounded-xl hover:border-primary-500/50 transition-all duration-300 group"
            >
              <Activity className="w-8 h-8 text-primary-400 mb-2 group-hover:scale-110 transition-transform" />
              <p className="font-medium">New Training</p>
              <p className="text-xs text-white/40 mt-1">Start a new PINN training job</p>
            </a>
            <a
              href="/visualization"
              className="p-4 bg-gradient-to-br from-accent-500/20 to-accent-600/10 border border-accent-500/30 rounded-xl hover:border-accent-500/50 transition-all duration-300 group"
            >
              <Zap className="w-8 h-8 text-accent-400 mb-2 group-hover:scale-110 transition-transform" />
              <p className="font-medium">Visualize</p>
              <p className="text-xs text-white/40 mt-1">View wavefields & models</p>
            </a>
            <a
              href="/models"
              className="p-4 bg-gradient-to-br from-green-500/20 to-green-600/10 border border-green-500/30 rounded-xl hover:border-green-500/50 transition-all duration-300 group"
            >
              <Box className="w-8 h-8 text-green-400 mb-2 group-hover:scale-110 transition-transform" />
              <p className="font-medium">Models</p>
              <p className="text-xs text-white/40 mt-1">Manage saved models</p>
            </a>
            <a
              href="/experiments"
              className="p-4 bg-gradient-to-br from-yellow-500/20 to-yellow-600/10 border border-yellow-500/30 rounded-xl hover:border-yellow-500/50 transition-all duration-300 group"
            >
              <BarChart3 className="w-8 h-8 text-yellow-400 mb-2 group-hover:scale-110 transition-transform" />
              <p className="font-medium">Experiments</p>
              <p className="text-xs text-white/40 mt-1">Compare & analyze results</p>
            </a>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}
