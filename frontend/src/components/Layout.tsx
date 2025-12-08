import { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Activity,
  Brain,
  BarChart3,
  Layers,
  FlaskConical,
  Settings,
  Waves,
  Radio,
  AlertTriangle,
  Map,
  Cpu,
  ChevronRight,
} from 'lucide-react';

interface LayoutProps {
  children: ReactNode;
}

const navigation = [
  { name: 'Dashboard', href: '/', icon: Activity },
  { name: 'Tsunami Map', href: '/tsunami', icon: AlertTriangle, highlight: true },
  { name: 'Training', href: '/training', icon: Brain },
  { name: 'Visualization', href: '/visualization', icon: Waves },
  { name: 'Models', href: '/models', icon: Layers },
  { name: 'Experiments', href: '/experiments', icon: FlaskConical },
  { name: 'Seismic Data', href: '/seismic', icon: Radio },
];

export default function Layout({ children }: LayoutProps) {
  const location = useLocation();

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

        .layout-container {
          min-height: 100vh;
          display: flex;
          background: #0a0a0a;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .sidebar {
          position: fixed;
          inset-y: 0;
          left: 0;
          width: 240px;
          background: #0d0d0d;
          border-right: 1px solid #1f1f1f;
          display: flex;
          flex-direction: column;
          z-index: 100;
        }

        .sidebar-header {
          padding: 20px;
          border-bottom: 1px solid #1f1f1f;
        }

        .logo {
          display: flex;
          align-items: center;
          gap: 12px;
          text-decoration: none;
          color: inherit;
        }

        .logo-icon {
          width: 40px;
          height: 40px;
          background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
          border-radius: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
        }

        .logo-text h1 {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 16px;
          font-weight: 700;
          color: #ffffff;
          letter-spacing: -0.5px;
        }

        .logo-text p {
          font-size: 10px;
          color: #525252;
          margin-top: 2px;
        }

        .sidebar-nav {
          flex: 1;
          padding: 16px 12px;
          overflow-y: auto;
        }

        .nav-item {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px 14px;
          margin-bottom: 4px;
          border-radius: 8px;
          text-decoration: none;
          color: #737373;
          font-size: 13px;
          font-weight: 500;
          transition: all 0.15s ease;
          position: relative;
        }

        .nav-item:hover {
          background: #1a1a1a;
          color: #e5e5e5;
        }

        .nav-item.active {
          background: rgba(59, 130, 246, 0.1);
          color: #60a5fa;
          border: 1px solid rgba(59, 130, 246, 0.2);
        }

        .nav-item.active::before {
          content: '';
          position: absolute;
          left: 0;
          top: 50%;
          transform: translateY(-50%);
          width: 3px;
          height: 20px;
          background: #3b82f6;
          border-radius: 0 2px 2px 0;
        }

        .nav-item.highlight {
          background: rgba(239, 68, 68, 0.08);
          color: #f87171;
          border: 1px solid rgba(239, 68, 68, 0.2);
        }

        .nav-item.highlight:hover {
          background: rgba(239, 68, 68, 0.12);
          border-color: rgba(239, 68, 68, 0.3);
        }

        .nav-item.highlight.active {
          background: rgba(239, 68, 68, 0.15);
          color: #fca5a5;
          border-color: rgba(239, 68, 68, 0.3);
        }

        .nav-item.highlight.active::before {
          background: #ef4444;
        }

        .nav-icon {
          width: 18px;
          height: 18px;
          flex-shrink: 0;
        }

        .nav-label {
          flex: 1;
        }

        .sidebar-footer {
          padding: 16px;
          border-top: 1px solid #1f1f1f;
        }

        .settings-btn {
          display: flex;
          align-items: center;
          gap: 12px;
          width: 100%;
          padding: 12px 14px;
          border: none;
          border-radius: 8px;
          background: transparent;
          color: #525252;
          font-size: 13px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.15s ease;
        }

        .settings-btn:hover {
          background: #1a1a1a;
          color: #e5e5e5;
        }

        .status-card {
          margin-top: 16px;
          padding: 14px;
          background: #141414;
          border: 1px solid #1f1f1f;
          border-radius: 8px;
        }

        .status-row {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .status-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #22c55e;
          animation: pulse 2s infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        .status-text {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 11px;
          color: #737373;
        }

        .status-sub {
          font-size: 10px;
          color: #404040;
          margin-top: 6px;
        }

        .main-content {
          flex: 1;
          margin-left: 240px;
          min-height: 100vh;
        }

        .content-wrapper {
          padding: 32px;
          min-height: 100vh;
        }
      `}</style>

      <div className="layout-container">
        {/* Sidebar */}
        <aside className="sidebar">
          {/* Logo */}
          <div className="sidebar-header">
            <Link to="/" className="logo">
              <div className="logo-icon">
                <Waves size={22} />
              </div>
              <div className="logo-text">
                <h1>PINN Seismic</h1>
                <p>Waveform Inversion</p>
              </div>
            </Link>
          </div>

          {/* Navigation */}
          <nav className="sidebar-nav">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;
              const Icon = item.icon;
              const isHighlight = 'highlight' in item && item.highlight;

              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`nav-item ${isActive ? 'active' : ''} ${isHighlight ? 'highlight' : ''}`}
                >
                  <Icon className="nav-icon" />
                  <span className="nav-label">{item.name}</span>
                </Link>
              );
            })}
          </nav>

          {/* Footer */}
          <div className="sidebar-footer">
            <button className="settings-btn">
              <Settings size={18} />
              <span>Settings</span>
            </button>

            <div className="status-card">
              <div className="status-row">
                <span className="status-dot" />
                <span className="status-text">API Connected</span>
              </div>
              <p className="status-sub">GPU: CUDA Available</p>
            </div>
          </div>
        </aside>

        {/* Main content */}
        <main className="main-content">
          <div className="content-wrapper">
            {children}
          </div>
        </main>
      </div>
    </>
  );
}
