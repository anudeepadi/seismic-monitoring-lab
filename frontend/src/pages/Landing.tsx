import { Link } from 'react-router-dom';
import { useState, useEffect } from 'react';

export default function Landing() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [stats, setStats] = useState({
    earthquakesToday: 47,
    activeStations: 12,
    tsunamiAlerts: 0,
    avgLatency: 2.3,
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(new Date());
      // Simulate live stats updates
      setStats(prev => ({
        ...prev,
        earthquakesToday: prev.earthquakesToday + (Math.random() > 0.8 ? 1 : 0),
        avgLatency: (2 + Math.random() * 0.8).toFixed(1) as unknown as number,
      }));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="landing-page">
      {/* Top Alert Bar */}
      <div className="alert-bar">
        <div className="alert-bar-content">
          <span className="status-indicator active" />
          <span className="alert-text">SYSTEM OPERATIONAL</span>
          <span className="separator">|</span>
          <span className="utc-time">UTC {currentTime.toISOString().slice(11, 19)}</span>
          <span className="separator">|</span>
          <span className="station-count">{stats.activeStations} STATIONS ONLINE</span>
        </div>
      </div>

      {/* Header */}
      <header className="main-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">
              <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="20" cy="20" r="18" stroke="currentColor" strokeWidth="2"/>
                <circle cx="20" cy="20" r="12" stroke="currentColor" strokeWidth="1.5"/>
                <circle cx="20" cy="20" r="6" stroke="currentColor" strokeWidth="1"/>
                <circle cx="20" cy="20" r="2" fill="currentColor"/>
                <path d="M20 2v6M20 32v6M2 20h6M32 20h6" stroke="currentColor" strokeWidth="1.5"/>
              </svg>
            </div>
            <div className="logo-text">
              <h1>SENTINEL</h1>
              <p>Seismic Event Network for Tsunami Intelligence</p>
            </div>
          </div>
          <nav className="main-nav">
            <Link to="/tsunami" className="nav-link">2D Map</Link>
            <Link to="/globe" className="nav-link">3D Globe</Link>
            <a href="#features" className="nav-link">Features</a>
            <a href="#data" className="nav-link">Live Data</a>
            <a href="#about" className="nav-link">About</a>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="hero">
        <div className="hero-background">
          <div className="wave-animation" />
        </div>
        <div className="hero-content">
          <div className="hero-badge">REAL-TIME MONITORING SYSTEM</div>
          <h2>Global Earthquake Detection<br />& Tsunami Warning Network</h2>
          <p>
            Advanced seismic monitoring platform providing real-time earthquake detection,
            wave propagation visualization, and tsunami early warning capabilities
            for coastal regions worldwide.
          </p>
          <div className="hero-actions">
            <Link to="/tsunami" className="btn-primary">
              <span>Launch 2D Map</span>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 12h14M12 5l7 7-7 7"/>
              </svg>
            </Link>
            <Link to="/globe" className="btn-secondary">
              <span>View 3D Globe</span>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"/>
                <ellipse cx="12" cy="12" rx="4" ry="10"/>
                <path d="M2 12h20"/>
              </svg>
            </Link>
          </div>
        </div>
        <div className="hero-stats">
          <div className="stat-card">
            <div className="stat-value">{stats.earthquakesToday}</div>
            <div className="stat-label">Earthquakes Today</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{stats.activeStations}</div>
            <div className="stat-label">Active Stations</div>
          </div>
          <div className="stat-card alert">
            <div className="stat-value">{stats.tsunamiAlerts}</div>
            <div className="stat-label">Active Tsunami Alerts</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{stats.avgLatency}s</div>
            <div className="stat-label">Avg Detection Time</div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="features-section">
        <div className="section-header">
          <span className="section-badge">CAPABILITIES</span>
          <h3>Advanced Detection & Warning System</h3>
          <p>Military-grade precision with real-time global coverage</p>
        </div>

        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon automatic">
              <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="24" cy="24" r="20"/>
                <path d="M24 4v8M24 36v8M4 24h8M36 24h8"/>
                <circle cx="24" cy="24" r="8" fill="currentColor" fillOpacity="0.2"/>
                <circle cx="24" cy="24" r="3"/>
              </svg>
            </div>
            <h4>Automatic Detection</h4>
            <p>
              Real-time seismograph monitoring with automatic event detection.
              Stations highlight in <span className="highlight-green">green</span> when
              increased activity is detected, providing instant visual alerts for
              operators and emergency responders.
            </p>
            <ul className="feature-list">
              <li>STA/LTA trigger algorithms</li>
              <li>Multi-station correlation</li>
              <li>Automatic noise filtering</li>
              <li>Sub-second detection latency</li>
            </ul>
          </div>

          <div className="feature-card">
            <div className="feature-icon propagation">
              <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="24" cy="24" r="6"/>
                <circle cx="24" cy="24" r="12" strokeDasharray="4 2"/>
                <circle cx="24" cy="24" r="18" strokeDasharray="4 2"/>
                <path d="M24 6v-2M24 44v-2M6 24H4M44 24h-2"/>
              </svg>
            </div>
            <h4>Wave Propagation Visualization</h4>
            <p>
              Live display of seismic wave fronts as they travel across the globe.
              Visualize both <span className="highlight-blue">P-waves</span> (primary) and
              <span className="highlight-orange"> S-waves</span> (secondary) in real-time
              with accurate travel time calculations.
            </p>
            <ul className="feature-list">
              <li>P-wave and S-wave tracking</li>
              <li>Tsunami wave propagation</li>
              <li>Geographic radius display</li>
              <li>Arrival time predictions</li>
            </ul>
          </div>

          <div className="feature-card">
            <div className="feature-icon data">
              <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="6" y="6" width="36" height="36" rx="4"/>
                <path d="M6 16h36M16 16v26"/>
                <path d="M22 24h14M22 32h10M22 40h6"/>
                <circle cx="11" cy="11" r="2" fill="currentColor"/>
              </svg>
            </div>
            <h4>Detailed Event Analysis</h4>
            <p>
              Comprehensive earthquake parameters calculated in real-time using
              advanced algorithms. Access magnitude, depth, location, intensity
              estimates, and potential tsunami threat assessment.
            </p>
            <ul className="feature-list">
              <li>Magnitude estimation (Mw, Ms, Mb)</li>
              <li>Hypocenter depth calculation</li>
              <li>Maximum intensity prediction</li>
              <li>Tsunami threat assessment</li>
            </ul>
          </div>

          <div className="feature-card">
            <div className="feature-icon warning">
              <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M24 6L42 42H6L24 6z"/>
                <path d="M24 18v12M24 34v2"/>
              </svg>
            </div>
            <h4>Tsunami Early Warning</h4>
            <p>
              Integrated tsunami warning system with coastal impact zone
              visualization. Automatic alerts for earthquakes meeting tsunami
              criteria: shallow depth (&lt;100km), high magnitude (&gt;7.0),
              and submarine location.
            </p>
            <ul className="feature-list">
              <li>Automatic tsunami assessment</li>
              <li>Coastal impact zones</li>
              <li>Wave arrival time estimates</li>
              <li>Audio-visual alerts</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Live Data Section */}
      <section id="data" className="data-section">
        <div className="section-header">
          <span className="section-badge">LIVE DATA</span>
          <h3>Global Monitoring Network</h3>
          <p>Connected to international seismic networks via IRIS SeedLink</p>
        </div>

        <div className="network-info">
          <div className="network-card">
            <h5>Data Sources</h5>
            <ul>
              <li><span className="network-badge iris">IRIS</span> Incorporated Research Institutions for Seismology</li>
              <li><span className="network-badge usgs">USGS</span> United States Geological Survey</li>
              <li><span className="network-badge geofon">GFZ</span> GEOFON Global Seismic Network</li>
              <li><span className="network-badge emsc">EMSC</span> European-Mediterranean Seismological Centre</li>
            </ul>
          </div>
          <div className="network-card">
            <h5>Coverage Regions</h5>
            <div className="regions-grid">
              <div className="region">
                <span className="region-status critical">CRITICAL</span>
                <span>Pacific Ring of Fire</span>
              </div>
              <div className="region">
                <span className="region-status high">HIGH</span>
                <span>Indian Ocean</span>
              </div>
              <div className="region">
                <span className="region-status high">HIGH</span>
                <span>Cascadia Subduction Zone</span>
              </div>
              <div className="region">
                <span className="region-status medium">MEDIUM</span>
                <span>Mediterranean</span>
              </div>
              <div className="region">
                <span className="region-status medium">MEDIUM</span>
                <span>Caribbean</span>
              </div>
              <div className="region">
                <span className="region-status low">LOW</span>
                <span>Atlantic Ocean</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="cta-content">
          <h3>Access the Live Monitoring System</h3>
          <p>Real-time earthquake detection and tsunami warning at your fingertips</p>
          <div className="cta-buttons">
            <Link to="/tsunami" className="btn-cta primary">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
                <path d="M3 9h18M9 21V9"/>
              </svg>
              <span>2D Map Interface</span>
            </Link>
            <Link to="/globe" className="btn-cta secondary">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"/>
                <ellipse cx="12" cy="12" rx="4" ry="10"/>
                <path d="M2 12h20"/>
              </svg>
              <span>3D Globe View</span>
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer id="about" className="main-footer">
        <div className="footer-content">
          <div className="footer-brand">
            <div className="footer-logo">
              <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="20" cy="20" r="18" stroke="currentColor" strokeWidth="2"/>
                <circle cx="20" cy="20" r="12" stroke="currentColor" strokeWidth="1.5"/>
                <circle cx="20" cy="20" r="6" stroke="currentColor" strokeWidth="1"/>
                <circle cx="20" cy="20" r="2" fill="currentColor"/>
              </svg>
              <span>SENTINEL</span>
            </div>
            <p>
              Seismic Event Network for Tsunami Intelligence and Early Listening.
              An advanced earthquake detection and tsunami warning system providing
              real-time monitoring capabilities for emergency preparedness.
            </p>
          </div>
          <div className="footer-links">
            <div className="footer-column">
              <h6>System</h6>
              <Link to="/tsunami">2D Map</Link>
              <Link to="/globe">3D Globe</Link>
              <Link to="/">Dashboard</Link>
            </div>
            <div className="footer-column">
              <h6>Data</h6>
              <a href="https://www.iris.edu" target="_blank" rel="noopener noreferrer">IRIS</a>
              <a href="https://earthquake.usgs.gov" target="_blank" rel="noopener noreferrer">USGS</a>
              <a href="https://www.emsc-csem.org" target="_blank" rel="noopener noreferrer">EMSC</a>
            </div>
            <div className="footer-column">
              <h6>Resources</h6>
              <a href="https://tsunami.gov" target="_blank" rel="noopener noreferrer">Tsunami.gov</a>
              <a href="https://www.weather.gov/tsunami" target="_blank" rel="noopener noreferrer">NWS Tsunami</a>
              <a href="https://www.ready.gov/tsunamis" target="_blank" rel="noopener noreferrer">Ready.gov</a>
            </div>
          </div>
        </div>
        <div className="footer-bottom">
          <p>SENTINEL Seismic Monitoring System | Data provided by international seismic networks</p>
          <p className="disclaimer">
            This system is for educational and research purposes. For official tsunami warnings,
            refer to your local emergency management authority.
          </p>
        </div>
      </footer>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        * {
          box-sizing: border-box;
          margin: 0;
          padding: 0;
        }

        .landing-page {
          min-height: 100vh;
          background: #030712;
          color: #e5e7eb;
          font-family: 'Inter', -apple-system, sans-serif;
        }

        /* Alert Bar */
        .alert-bar {
          background: linear-gradient(90deg, #065f46 0%, #047857 50%, #065f46 100%);
          padding: 8px 0;
          border-bottom: 1px solid #10b981;
        }

        .alert-bar-content {
          max-width: 1400px;
          margin: 0 auto;
          padding: 0 24px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 16px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 12px;
          font-weight: 500;
          color: #a7f3d0;
        }

        .status-indicator {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #10b981;
          animation: pulse 2s infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
          50% { opacity: 0.8; box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        }

        .separator {
          color: #6ee7b7;
          opacity: 0.4;
        }

        /* Header */
        .main-header {
          background: rgba(3, 7, 18, 0.95);
          backdrop-filter: blur(12px);
          border-bottom: 1px solid #1f2937;
          position: sticky;
          top: 0;
          z-index: 100;
        }

        .header-content {
          max-width: 1400px;
          margin: 0 auto;
          padding: 16px 24px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .logo-section {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .logo-icon {
          width: 48px;
          height: 48px;
          color: #10b981;
        }

        .logo-text h1 {
          font-size: 24px;
          font-weight: 800;
          letter-spacing: 4px;
          color: #f9fafb;
          margin: 0;
        }

        .logo-text p {
          font-size: 10px;
          color: #6b7280;
          letter-spacing: 1px;
          text-transform: uppercase;
          margin: 0;
        }

        .main-nav {
          display: flex;
          gap: 8px;
        }

        .nav-link {
          padding: 10px 20px;
          color: #9ca3af;
          text-decoration: none;
          font-size: 14px;
          font-weight: 500;
          border-radius: 6px;
          transition: all 0.2s;
        }

        .nav-link:hover {
          color: #f9fafb;
          background: #1f2937;
        }

        /* Hero */
        .hero {
          position: relative;
          padding: 100px 24px 60px;
          overflow: hidden;
        }

        .hero-background {
          position: absolute;
          inset: 0;
          background: radial-gradient(ellipse at top, rgba(16, 185, 129, 0.1) 0%, transparent 50%);
        }

        .wave-animation {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 800px;
          height: 800px;
          border: 1px solid rgba(16, 185, 129, 0.1);
          border-radius: 50%;
          animation: waveExpand 4s infinite ease-out;
        }

        .wave-animation::before,
        .wave-animation::after {
          content: '';
          position: absolute;
          inset: -100px;
          border: 1px solid rgba(16, 185, 129, 0.05);
          border-radius: 50%;
          animation: waveExpand 4s infinite ease-out;
        }

        .wave-animation::before { animation-delay: 1.5s; }
        .wave-animation::after { animation-delay: 3s; }

        @keyframes waveExpand {
          0% { transform: scale(0.5); opacity: 1; }
          100% { transform: scale(2); opacity: 0; }
        }

        .hero-content {
          position: relative;
          max-width: 800px;
          margin: 0 auto;
          text-align: center;
        }

        .hero-badge {
          display: inline-block;
          padding: 8px 16px;
          background: rgba(16, 185, 129, 0.1);
          border: 1px solid rgba(16, 185, 129, 0.3);
          border-radius: 100px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
          font-weight: 600;
          letter-spacing: 2px;
          color: #10b981;
          margin-bottom: 24px;
        }

        .hero-content h2 {
          font-size: 56px;
          font-weight: 800;
          line-height: 1.1;
          color: #f9fafb;
          margin-bottom: 24px;
        }

        .hero-content p {
          font-size: 18px;
          line-height: 1.7;
          color: #9ca3af;
          margin-bottom: 40px;
        }

        .hero-actions {
          display: flex;
          justify-content: center;
          gap: 16px;
          margin-bottom: 80px;
        }

        .btn-primary, .btn-secondary {
          display: inline-flex;
          align-items: center;
          gap: 10px;
          padding: 16px 32px;
          font-size: 15px;
          font-weight: 600;
          border-radius: 8px;
          text-decoration: none;
          transition: all 0.2s;
        }

        .btn-primary {
          background: linear-gradient(135deg, #059669 0%, #10b981 100%);
          color: white;
          box-shadow: 0 4px 24px rgba(16, 185, 129, 0.3);
        }

        .btn-primary:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 32px rgba(16, 185, 129, 0.4);
        }

        .btn-primary svg, .btn-secondary svg {
          width: 20px;
          height: 20px;
        }

        .btn-secondary {
          background: #1f2937;
          color: #e5e7eb;
          border: 1px solid #374151;
        }

        .btn-secondary:hover {
          background: #374151;
          border-color: #4b5563;
        }

        .hero-stats {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 24px;
          max-width: 1000px;
          margin: 0 auto;
        }

        .stat-card {
          background: rgba(31, 41, 55, 0.5);
          border: 1px solid #374151;
          border-radius: 12px;
          padding: 24px;
          text-align: center;
        }

        .stat-card.alert {
          border-color: #10b981;
          background: rgba(16, 185, 129, 0.05);
        }

        .stat-value {
          font-family: 'JetBrains Mono', monospace;
          font-size: 36px;
          font-weight: 700;
          color: #f9fafb;
        }

        .stat-card.alert .stat-value {
          color: #10b981;
        }

        .stat-label {
          font-size: 13px;
          color: #6b7280;
          margin-top: 8px;
        }

        /* Features Section */
        .features-section {
          padding: 100px 24px;
          background: #0a0f1a;
        }

        .section-header {
          text-align: center;
          max-width: 600px;
          margin: 0 auto 60px;
        }

        .section-badge {
          display: inline-block;
          padding: 6px 14px;
          background: rgba(59, 130, 246, 0.1);
          border: 1px solid rgba(59, 130, 246, 0.3);
          border-radius: 100px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
          font-weight: 600;
          letter-spacing: 2px;
          color: #3b82f6;
          margin-bottom: 16px;
        }

        .section-header h3 {
          font-size: 36px;
          font-weight: 700;
          color: #f9fafb;
          margin-bottom: 12px;
        }

        .section-header p {
          font-size: 16px;
          color: #6b7280;
        }

        .features-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 24px;
          max-width: 1200px;
          margin: 0 auto;
        }

        .feature-card {
          background: linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%);
          border: 1px solid #374151;
          border-radius: 16px;
          padding: 32px;
          transition: all 0.3s;
        }

        .feature-card:hover {
          border-color: #4b5563;
          transform: translateY(-4px);
        }

        .feature-icon {
          width: 64px;
          height: 64px;
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 20px;
        }

        .feature-icon svg {
          width: 36px;
          height: 36px;
        }

        .feature-icon.automatic {
          background: rgba(16, 185, 129, 0.1);
          color: #10b981;
        }

        .feature-icon.propagation {
          background: rgba(59, 130, 246, 0.1);
          color: #3b82f6;
        }

        .feature-icon.data {
          background: rgba(168, 85, 247, 0.1);
          color: #a855f7;
        }

        .feature-icon.warning {
          background: rgba(239, 68, 68, 0.1);
          color: #ef4444;
        }

        .feature-card h4 {
          font-size: 20px;
          font-weight: 600;
          color: #f9fafb;
          margin-bottom: 12px;
        }

        .feature-card p {
          font-size: 14px;
          line-height: 1.7;
          color: #9ca3af;
          margin-bottom: 20px;
        }

        .highlight-green { color: #10b981; font-weight: 600; }
        .highlight-blue { color: #3b82f6; font-weight: 600; }
        .highlight-orange { color: #f97316; font-weight: 600; }

        .feature-list {
          list-style: none;
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
        }

        .feature-list li {
          font-size: 13px;
          color: #6b7280;
          padding-left: 20px;
          position: relative;
        }

        .feature-list li::before {
          content: '→';
          position: absolute;
          left: 0;
          color: #4b5563;
        }

        /* Data Section */
        .data-section {
          padding: 100px 24px;
          background: #030712;
        }

        .network-info {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 24px;
          max-width: 1200px;
          margin: 0 auto;
        }

        .network-card {
          background: #0a0f1a;
          border: 1px solid #1f2937;
          border-radius: 12px;
          padding: 32px;
        }

        .network-card h5 {
          font-size: 14px;
          font-weight: 600;
          color: #9ca3af;
          text-transform: uppercase;
          letter-spacing: 1px;
          margin-bottom: 20px;
        }

        .network-card ul {
          list-style: none;
        }

        .network-card li {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px 0;
          border-bottom: 1px solid #1f2937;
          font-size: 14px;
          color: #e5e7eb;
        }

        .network-card li:last-child {
          border-bottom: none;
        }

        .network-badge {
          padding: 4px 10px;
          border-radius: 4px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          font-weight: 700;
        }

        .network-badge.iris { background: #1e3a5f; color: #60a5fa; }
        .network-badge.usgs { background: #3f1e1e; color: #f87171; }
        .network-badge.geofon { background: #1e3f1e; color: #4ade80; }
        .network-badge.emsc { background: #3f3f1e; color: #fbbf24; }

        .regions-grid {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .region {
          display: flex;
          align-items: center;
          gap: 12px;
          font-size: 14px;
          color: #e5e7eb;
        }

        .region-status {
          padding: 4px 10px;
          border-radius: 4px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          font-weight: 700;
          min-width: 70px;
          text-align: center;
        }

        .region-status.critical { background: #7f1d1d; color: #fca5a5; }
        .region-status.high { background: #78350f; color: #fcd34d; }
        .region-status.medium { background: #1e3a5f; color: #93c5fd; }
        .region-status.low { background: #14532d; color: #86efac; }

        /* CTA Section */
        .cta-section {
          padding: 100px 24px;
          background: linear-gradient(180deg, #0a0f1a 0%, #030712 100%);
          text-align: center;
        }

        .cta-content {
          max-width: 600px;
          margin: 0 auto;
        }

        .cta-content h3 {
          font-size: 32px;
          font-weight: 700;
          color: #f9fafb;
          margin-bottom: 12px;
        }

        .cta-content p {
          font-size: 16px;
          color: #6b7280;
          margin-bottom: 32px;
        }

        .cta-buttons {
          display: flex;
          justify-content: center;
          gap: 16px;
        }

        .btn-cta {
          display: inline-flex;
          align-items: center;
          gap: 12px;
          padding: 20px 40px;
          font-size: 16px;
          font-weight: 600;
          border-radius: 10px;
          text-decoration: none;
          transition: all 0.2s;
        }

        .btn-cta svg {
          width: 24px;
          height: 24px;
        }

        .btn-cta.primary {
          background: linear-gradient(135deg, #059669 0%, #10b981 100%);
          color: white;
          box-shadow: 0 4px 24px rgba(16, 185, 129, 0.3);
        }

        .btn-cta.primary:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 32px rgba(16, 185, 129, 0.4);
        }

        .btn-cta.secondary {
          background: #1f2937;
          color: #e5e7eb;
          border: 1px solid #374151;
        }

        .btn-cta.secondary:hover {
          background: #374151;
        }

        /* Footer */
        .main-footer {
          background: #030712;
          border-top: 1px solid #1f2937;
          padding: 60px 24px 30px;
        }

        .footer-content {
          display: grid;
          grid-template-columns: 2fr 3fr;
          gap: 60px;
          max-width: 1200px;
          margin: 0 auto;
        }

        .footer-brand {
          max-width: 350px;
        }

        .footer-logo {
          display: flex;
          align-items: center;
          gap: 12px;
          color: #f9fafb;
          margin-bottom: 16px;
        }

        .footer-logo svg {
          width: 32px;
          height: 32px;
          color: #10b981;
        }

        .footer-logo span {
          font-size: 20px;
          font-weight: 800;
          letter-spacing: 3px;
        }

        .footer-brand p {
          font-size: 14px;
          line-height: 1.7;
          color: #6b7280;
        }

        .footer-links {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 40px;
        }

        .footer-column h6 {
          font-size: 12px;
          font-weight: 600;
          color: #9ca3af;
          text-transform: uppercase;
          letter-spacing: 1px;
          margin-bottom: 16px;
        }

        .footer-column a {
          display: block;
          padding: 6px 0;
          color: #6b7280;
          text-decoration: none;
          font-size: 14px;
          transition: color 0.2s;
        }

        .footer-column a:hover {
          color: #f9fafb;
        }

        .footer-bottom {
          max-width: 1200px;
          margin: 40px auto 0;
          padding-top: 24px;
          border-top: 1px solid #1f2937;
          text-align: center;
        }

        .footer-bottom p {
          font-size: 13px;
          color: #4b5563;
        }

        .disclaimer {
          margin-top: 8px;
          font-style: italic;
        }

        /* Responsive */
        @media (max-width: 1024px) {
          .features-grid {
            grid-template-columns: 1fr;
          }

          .network-info {
            grid-template-columns: 1fr;
          }

          .hero-stats {
            grid-template-columns: repeat(2, 1fr);
          }
        }

        @media (max-width: 768px) {
          .header-content {
            flex-direction: column;
            gap: 16px;
          }

          .main-nav {
            flex-wrap: wrap;
            justify-content: center;
          }

          .hero-content h2 {
            font-size: 36px;
          }

          .hero-actions {
            flex-direction: column;
          }

          .hero-stats {
            grid-template-columns: 1fr;
          }

          .cta-buttons {
            flex-direction: column;
          }

          .footer-content {
            grid-template-columns: 1fr;
          }

          .footer-links {
            grid-template-columns: 1fr;
            gap: 24px;
          }
        }
      `}</style>
    </div>
  );
}
