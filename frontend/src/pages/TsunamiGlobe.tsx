import { useState, useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import CesiumGlobe from '../components/CesiumGlobe';
import { useSeismicStream, SeismicEvent as APIEvent } from '../hooks/useSeismicStream';

// Indian Ocean monitoring stations
const STATIONS = [
  { id: 'PALK', network: 'II', name: 'Pallekele', country: 'Sri Lanka', lat: 7.2728, lng: 80.7022, status: 'online' },
  { id: 'COCO', network: 'II', name: 'Cocos Islands', country: 'Australia', lat: -12.1901, lng: 96.8349, status: 'online' },
  { id: 'DGAR', network: 'II', name: 'Diego Garcia', country: 'BIOT', lat: -7.4121, lng: 72.4525, status: 'online' },
  { id: 'CHTO', network: 'IU', name: 'Chiang Mai', country: 'Thailand', lat: 18.8141, lng: 98.9443, status: 'online' },
  { id: 'TATO', network: 'IU', name: 'Taipei', country: 'Taiwan', lat: 24.9735, lng: 121.4971, status: 'online' },
  { id: 'NWAO', network: 'IU', name: 'Narrogin', country: 'Australia', lat: -32.9277, lng: 117.2390, status: 'online' },
  { id: 'WRAB', network: 'II', name: 'Warramunga', country: 'Australia', lat: -19.9336, lng: 134.3600, status: 'online' },
  { id: 'MBWA', network: 'IU', name: 'Marble Bar', country: 'Australia', lat: -21.1590, lng: 119.7313, status: 'online' },
];

interface Station {
  id: string;
  network: string;
  name: string;
  country: string;
  lat: number;
  lng: number;
  status: string;
  amplitude?: number;
}

interface RiskAssessment {
  level: 'OK' | 'WATCH' | 'WARNING' | 'CRITICAL';
  indiaStatus: string;
  threateningEvents: APIEvent[];
  estimatedArrivalTime: string | null;
  affectedRegions: string[];
}

export default function TsunamiGlobe() {
  const { stations: apiStations, events: liveEvents, indianOceanEvents, connected } = useSeismicStream();

  const [stations, setStations] = useState<Station[]>(STATIONS.map(s => ({ ...s, amplitude: 0.2 })));
  const [selectedStation, setSelectedStation] = useState<Station | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<APIEvent | null>(null);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [riskAssessment, setRiskAssessment] = useState<RiskAssessment>({
    level: 'OK',
    indiaStatus: 'NO IMMEDIATE THREAT',
    threateningEvents: [],
    estimatedArrivalTime: null,
    affectedRegions: [],
  });
  const [sirenPlaying, setSirenPlaying] = useState(false);
  const [simulationActive, setSimulationActive] = useState(false);
  const [simulatedEvent, setSimulatedEvent] = useState<APIEvent | null>(null);
  const [waveRadius, setWaveRadius] = useState(0);
  const [simulationTime, setSimulationTime] = useState(0);
  const [simMenuOpen, setSimMenuOpen] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const oscillatorsRef = useRef<OscillatorNode[]>([]);
  const simulationIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Emergency Alert Siren
  const playSiren = useCallback(() => {
    if (sirenPlaying) return;

    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioContext;

      const masterGain = audioContext.createGain();
      masterGain.connect(audioContext.destination);
      masterGain.gain.value = 0.4;

      const now = audioContext.currentTime;
      const duration = 30;

      // Main siren sweep
      const mainOsc = audioContext.createOscillator();
      const mainGain = audioContext.createGain();
      mainOsc.connect(mainGain);
      mainGain.connect(masterGain);
      mainOsc.type = 'sawtooth';
      mainGain.gain.value = 0.5;

      for (let i = 0; i < 10; i++) {
        const cycleStart = now + i * 3;
        mainOsc.frequency.setValueAtTime(120, cycleStart);
        mainOsc.frequency.linearRampToValueAtTime(480, cycleStart + 2);
        mainOsc.frequency.setValueAtTime(480, cycleStart + 2.5);
        mainOsc.frequency.linearRampToValueAtTime(120, cycleStart + 2.8);
      }

      // Sub bass
      const subOsc = audioContext.createOscillator();
      const subGain = audioContext.createGain();
      subOsc.connect(subGain);
      subGain.connect(masterGain);
      subOsc.type = 'sine';
      subOsc.frequency.value = 55;
      subGain.gain.value = 0.3;

      mainOsc.start(now);
      subOsc.start(now);
      mainOsc.stop(now + duration);
      subOsc.stop(now + duration);

      oscillatorsRef.current = [mainOsc, subOsc];
      setSirenPlaying(true);
      setTimeout(() => setSirenPlaying(false), duration * 1000);
    } catch (e) {
      console.error('Audio playback failed:', e);
    }
  }, [sirenPlaying]);

  const stopSiren = useCallback(() => {
    oscillatorsRef.current.forEach(osc => {
      try { osc.stop(); } catch (e) {}
    });
    oscillatorsRef.current = [];
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setSirenPlaying(false);
  }, []);

  // Simulation scenarios
  const SIMULATION_SCENARIOS = [
    {
      name: '2004 Sumatra Earthquake',
      event: {
        event_id: 'sim_sumatra_2004',
        timestamp: new Date().toISOString(),
        lat: 3.316,
        lng: 95.854,
        magnitude: 9.1,
        depth: 30,
        place: 'Off the west coast of Sumatra, Indonesia',
        tsunami: true,
        in_region: true,
      },
    },
    {
      name: 'Andaman Mega Event',
      event: {
        event_id: 'sim_andaman_mega',
        timestamp: new Date().toISOString(),
        lat: 10.5,
        lng: 92.5,
        magnitude: 8.9,
        depth: 25,
        place: 'Andaman Sea, near Nicobar Islands',
        tsunami: true,
        in_region: true,
      },
    },
    {
      name: 'Bay of Bengal Event',
      event: {
        event_id: 'sim_bay_bengal',
        timestamp: new Date().toISOString(),
        lat: 14.5,
        lng: 87.0,
        magnitude: 7.8,
        depth: 15,
        place: 'Bay of Bengal, 400km east of Chennai',
        tsunami: true,
        in_region: true,
      },
    },
  ];

  const startSimulation = useCallback((scenarioIndex: number = 0) => {
    if (simulationActive) return;

    const scenario = SIMULATION_SCENARIOS[scenarioIndex];
    setSimulationActive(true);
    setSimulatedEvent(scenario.event as APIEvent);
    setWaveRadius(0);
    setSimulationTime(0);
    playSiren();

    const startTime = Date.now();
    simulationIntervalRef.current = setInterval(() => {
      const elapsed = (Date.now() - startTime) / 1000;
      const simulatedMinutes = elapsed * 10;
      const waveDistanceKm = (simulatedMinutes / 60) * 700;

      setSimulationTime(Math.round(simulatedMinutes));
      setWaveRadius(waveDistanceKm);

      if (simulatedMinutes > 240) {
        stopSimulation();
      }
    }, 100);
  }, [simulationActive, playSiren]);

  const stopSimulation = useCallback(() => {
    if (simulationIntervalRef.current) {
      clearInterval(simulationIntervalRef.current);
      simulationIntervalRef.current = null;
    }
    setSimulationActive(false);
    setSimulatedEvent(null);
    setWaveRadius(0);
    setSimulationTime(0);
    stopSiren();
  }, [stopSiren]);

  useEffect(() => {
    return () => {
      if (simulationIntervalRef.current) {
        clearInterval(simulationIntervalRef.current);
      }
    };
  }, []);

  const calculateDistanceToIndia = (lat: number, lng: number): number => {
    const indiaLat = 13.0;
    const indiaLng = 80.0;
    const R = 6371;
    const dLat = (indiaLat - lat) * Math.PI / 180;
    const dLng = (indiaLng - lng) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat * Math.PI / 180) * Math.cos(indiaLat * Math.PI / 180) *
              Math.sin(dLng/2) * Math.sin(dLng/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  };

  // Risk assessment
  useEffect(() => {
    const allEvents = simulatedEvent ? [...liveEvents, simulatedEvent] : liveEvents;
    const allIndianOceanEvents = simulatedEvent && simulatedEvent.in_region
      ? [...indianOceanEvents, simulatedEvent]
      : indianOceanEvents;

    const tsunamiEvents = allEvents.filter(e => e.tsunami);
    const highMagEvents = allEvents.filter(e => e.magnitude >= 7.0 && e.depth < 100);

    let level: RiskAssessment['level'] = 'OK';
    let indiaStatus = 'NO IMMEDIATE THREAT';
    let affectedRegions: string[] = [];
    let estimatedArrivalTime: string | null = null;
    let threateningEvents: APIEvent[] = [];

    if (tsunamiEvents.length > 0) {
      const nearestTsunami = tsunamiEvents.reduce((nearest, event) => {
        const dist = calculateDistanceToIndia(event.lat, event.lng);
        return dist < calculateDistanceToIndia(nearest.lat, nearest.lng) ? event : nearest;
      }, tsunamiEvents[0]);

      const distance = calculateDistanceToIndia(nearestTsunami.lat, nearestTsunami.lng);
      const tsunamiSpeed = 700;
      const arrivalHours = distance / tsunamiSpeed;

      if (distance < 3000) {
        level = 'CRITICAL';
        indiaStatus = 'TSUNAMI WARNING - EVACUATE COASTAL AREAS';
        estimatedArrivalTime = `${Math.round(arrivalHours * 60)} MINUTES`;
        affectedRegions = ['Tamil Nadu', 'Andhra Pradesh', 'Andaman & Nicobar', 'Kerala'];
        threateningEvents = tsunamiEvents;
        playSiren();
      }
    } else if (highMagEvents.length > 0) {
      const nearestEvent = highMagEvents.reduce((nearest, event) => {
        const dist = calculateDistanceToIndia(event.lat, event.lng);
        return dist < calculateDistanceToIndia(nearest.lat, nearest.lng) ? event : nearest;
      }, highMagEvents[0]);

      const distance = calculateDistanceToIndia(nearestEvent.lat, nearestEvent.lng);

      if (distance < 2000 && nearestEvent.magnitude >= 7.5) {
        level = 'WARNING';
        indiaStatus = 'HIGH SEISMIC ACTIVITY - MONITOR ALERTS';
        affectedRegions = ['Tamil Nadu', 'Andaman & Nicobar'];
        threateningEvents = highMagEvents;
      } else if (distance < 4000) {
        level = 'WATCH';
        indiaStatus = 'ELEVATED SEISMIC ACTIVITY';
        threateningEvents = highMagEvents;
      }
    }

    setRiskAssessment({ level, indiaStatus, threateningEvents, estimatedArrivalTime, affectedRegions });
  }, [liveEvents, indianOceanEvents, simulatedEvent, playSiren]);

  // Update time and amplitudes
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(new Date());
      if (apiStations.length > 0) {
        setStations(apiStations.map(s => ({
          id: s.id,
          network: s.network,
          name: s.name,
          country: s.country,
          lat: s.lat,
          lng: s.lng,
          status: s.status,
          amplitude: s.amplitude ?? Math.random() * 0.5 + 0.1,
        })));
      } else {
        setStations(prev => prev.map(s => ({
          ...s,
          amplitude: Math.random() * 0.5 + 0.1,
        })));
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [apiStations]);

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'CRITICAL': return '#dc2626';
      case 'WARNING': return '#f59e0b';
      case 'WATCH': return '#3b82f6';
      default: return '#10b981';
    }
  };

  return (
    <div className="tsunami-globe-system">
      {/* Compact Header */}
      <header className="globe-header">
        <div className="header-brand">
          <span className="brand-icon">&#9678;</span>
          <div className="brand-text">
            <span className="brand-title">TSUNAMI EARLY WARNING</span>
            <span className="brand-sub">3D GLOBE VIEW | INDIAN OCEAN</span>
          </div>
        </div>

        <div className="header-clock">
          <span className="clock-utc">UTC</span>
          <span className="clock-time">{currentTime.toUTCString().slice(17, 25)}</span>
        </div>

        <div className="header-status">
          <div className={`status-badge ${connected ? 'online' : 'offline'}`}>
            <span className="status-dot" />
            {connected ? 'LIVE' : 'OFFLINE'}
          </div>
          <div className="stat-pills">
            <span className="stat-pill">{liveEvents.length} Events</span>
            <span className="stat-pill highlight">{indianOceanEvents.length} IO</span>
          </div>
        </div>
      </header>

      {/* Alert Banner */}
      <div className={`globe-alert ${riskAssessment.level.toLowerCase()}`}>
        <div className="alert-left">
          <span className={`alert-indicator ${riskAssessment.level.toLowerCase()}`} />
          <span className="alert-label">INDIA:</span>
          <span className="alert-text" style={{ color: getRiskColor(riskAssessment.level) }}>
            {riskAssessment.level === 'OK' ? 'ALL CLEAR' : riskAssessment.indiaStatus}
          </span>
          {riskAssessment.estimatedArrivalTime && (
            <span className="alert-eta">ETA: {riskAssessment.estimatedArrivalTime}</span>
          )}
        </div>

        <div className="alert-right">
          {sirenPlaying && (
            <button className="siren-stop" onClick={stopSiren}>STOP ALERT</button>
          )}

          {!simulationActive ? (
            <div className={`sim-dropdown ${simMenuOpen ? 'open' : ''}`}>
              <button className="sim-btn" onClick={() => setSimMenuOpen(!simMenuOpen)}>
                &#9654; SIMULATE
              </button>
              {simMenuOpen && (
                <div className="sim-menu">
                  {SIMULATION_SCENARIOS.map((scenario, idx) => (
                    <button
                      key={idx}
                      className="sim-option"
                      onClick={() => { startSimulation(idx); setSimMenuOpen(false); }}
                    >
                      <span className="sim-mag">M{scenario.event.magnitude}</span>
                      <span className="sim-name">{scenario.name}</span>
                    </button>
                  ))}
                  <button className="sim-cancel" onClick={() => setSimMenuOpen(false)}>Cancel</button>
                </div>
              )}
            </div>
          ) : (
            <div className="sim-active-bar">
              <span className="sim-badge">SIM</span>
              <span className="sim-time">T+{Math.floor(simulationTime / 60)}h {simulationTime % 60}m</span>
              <span className="sim-wave">{Math.round(waveRadius)}km</span>
              <button className="sim-stop" onClick={stopSimulation}>&#9632; STOP</button>
            </div>
          )}

          <a href="/tsunami" className="view-switch">2D MAP</a>
        </div>
      </div>

      {/* Main Content */}
      <div className="globe-main">
        {/* 3D Globe */}
        <div className="cesium-container">
          <CesiumGlobe
            stations={stations}
            events={liveEvents}
            simulatedEvent={simulatedEvent}
            waveRadius={waveRadius}
            onStationClick={setSelectedStation}
            onEventClick={setSelectedEvent}
          />
        </div>

        {/* Side Panel */}
        <aside className="globe-sidebar">
          {/* Station Info */}
          {selectedStation && (
            <motion.div
              className="info-card station-card"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <div className="card-header">
                <span className="card-badge">{selectedStation.network}</span>
                <span className="card-title">{selectedStation.id}</span>
                <button className="card-close" onClick={() => setSelectedStation(null)}>&times;</button>
              </div>
              <div className="card-body">
                <div className="card-row">
                  <span className="row-label">Location</span>
                  <span className="row-value">{selectedStation.name}, {selectedStation.country}</span>
                </div>
                <div className="card-row">
                  <span className="row-label">Coordinates</span>
                  <span className="row-value">{selectedStation.lat.toFixed(3)}°, {selectedStation.lng.toFixed(3)}°</span>
                </div>
                <div className="card-row">
                  <span className="row-label">Status</span>
                  <span className={`row-value status-${selectedStation.status}`}>
                    {selectedStation.status.toUpperCase()}
                  </span>
                </div>
                <div className="card-row">
                  <span className="row-label">Amplitude</span>
                  <div className="amplitude-bar">
                    <div
                      className="amplitude-fill"
                      style={{ width: `${(selectedStation.amplitude || 0) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Event Info */}
          {selectedEvent && (
            <motion.div
              className={`info-card event-card ${selectedEvent.tsunami ? 'tsunami' : ''}`}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <div className="card-header">
                <span className="card-mag">M{selectedEvent.magnitude.toFixed(1)}</span>
                {selectedEvent.tsunami && <span className="tsunami-tag">TSUNAMI</span>}
                <button className="card-close" onClick={() => setSelectedEvent(null)}>&times;</button>
              </div>
              <div className="card-body">
                <div className="card-row">
                  <span className="row-label">Location</span>
                  <span className="row-value">{selectedEvent.place}</span>
                </div>
                <div className="card-row">
                  <span className="row-label">Depth</span>
                  <span className="row-value">{selectedEvent.depth.toFixed(1)} km</span>
                </div>
                <div className="card-row">
                  <span className="row-label">Time</span>
                  <span className="row-value">{new Date(selectedEvent.timestamp).toLocaleString()}</span>
                </div>
                <div className="card-row">
                  <span className="row-label">Distance to India</span>
                  <span className="row-value">{Math.round(calculateDistanceToIndia(selectedEvent.lat, selectedEvent.lng))} km</span>
                </div>
              </div>
            </motion.div>
          )}

          {/* Recent Events List */}
          <div className="events-panel">
            <div className="panel-header">
              <span className="panel-title">RECENT EVENTS</span>
              <span className="panel-count">{liveEvents.length}</span>
            </div>
            <div className="events-scroll">
              {liveEvents.slice(0, 15).map(event => (
                <div
                  key={event.event_id}
                  className={`event-row ${event.tsunami ? 'tsunami' : ''} ${event.in_region ? 'io' : ''}`}
                  onClick={() => setSelectedEvent(event)}
                >
                  <span className={`event-mag ${event.magnitude >= 6 ? 'high' : event.magnitude >= 5 ? 'med' : ''}`}>
                    {event.magnitude.toFixed(1)}
                  </span>
                  <div className="event-info">
                    <span className="event-place">{event.place}</span>
                    <span className="event-meta">{event.depth.toFixed(0)}km | {new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                  </div>
                  {event.in_region && <span className="io-tag">IO</span>}
                </div>
              ))}
            </div>
          </div>

          {/* Stations List */}
          <div className="stations-panel">
            <div className="panel-header">
              <span className="panel-title">STATIONS</span>
              <span className="panel-count">{stations.filter(s => s.status === 'online').length}/{stations.length}</span>
            </div>
            <div className="stations-grid">
              {stations.map(station => (
                <div
                  key={station.id}
                  className={`station-chip ${station.status}`}
                  onClick={() => setSelectedStation(station)}
                >
                  <span className="chip-dot" />
                  <span className="chip-id">{station.id}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        * {
          box-sizing: border-box;
        }

        .tsunami-globe-system {
          width: 100%;
          height: 100vh;
          background: #000;
          font-family: 'Inter', sans-serif;
          color: #e5e5e5;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        /* Header */
        .globe-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 10px 16px;
          background: rgba(10, 10, 10, 0.95);
          border-bottom: 1px solid #222;
          z-index: 100;
        }

        .header-brand {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .brand-icon {
          font-size: 20px;
          color: #10b981;
        }

        .brand-text {
          display: flex;
          flex-direction: column;
        }

        .brand-title {
          font-size: 13px;
          font-weight: 700;
          letter-spacing: 1px;
          color: #fff;
        }

        .brand-sub {
          font-size: 9px;
          color: #666;
          letter-spacing: 0.5px;
        }

        .header-clock {
          display: flex;
          align-items: baseline;
          gap: 8px;
        }

        .clock-utc {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          color: #525252;
        }

        .clock-time {
          font-family: 'JetBrains Mono', monospace;
          font-size: 20px;
          font-weight: 600;
          color: #fff;
          letter-spacing: 2px;
        }

        .header-status {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .status-badge {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 5px 10px;
          border-radius: 4px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          font-weight: 600;
        }

        .status-badge.online {
          background: rgba(16, 185, 129, 0.15);
          border: 1px solid rgba(16, 185, 129, 0.3);
          color: #10b981;
        }

        .status-badge.offline {
          background: rgba(239, 68, 68, 0.15);
          border: 1px solid rgba(239, 68, 68, 0.3);
          color: #ef4444;
        }

        .status-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: currentColor;
          animation: blink 1.5s infinite;
        }

        @keyframes blink {
          50% { opacity: 0.4; }
        }

        .stat-pills {
          display: flex;
          gap: 8px;
        }

        .stat-pill {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          padding: 4px 8px;
          background: #1a1a1a;
          border-radius: 3px;
          color: #888;
        }

        .stat-pill.highlight {
          background: rgba(59, 130, 246, 0.15);
          color: #60a5fa;
        }

        /* Alert Banner */
        .globe-alert {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 16px;
          background: #0d0d0d;
          border-bottom: 1px solid #222;
        }

        .globe-alert.critical {
          background: rgba(220, 38, 38, 0.1);
          animation: alertPulse 1s infinite;
        }

        @keyframes alertPulse {
          50% { background: rgba(220, 38, 38, 0.2); }
        }

        .alert-left {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .alert-indicator {
          width: 10px;
          height: 10px;
          border-radius: 50%;
        }

        .alert-indicator.ok { background: #10b981; }
        .alert-indicator.watch { background: #3b82f6; }
        .alert-indicator.warning { background: #f59e0b; animation: pulse 1s infinite; }
        .alert-indicator.critical { background: #dc2626; animation: pulse 0.5s infinite; }

        @keyframes pulse {
          50% { box-shadow: 0 0 0 6px transparent; }
        }

        .alert-label {
          font-size: 11px;
          font-weight: 600;
          color: #666;
        }

        .alert-text {
          font-family: 'JetBrains Mono', monospace;
          font-size: 12px;
          font-weight: 600;
        }

        .alert-eta {
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
          color: #dc2626;
          padding: 2px 8px;
          background: rgba(220, 38, 38, 0.2);
          border-radius: 3px;
        }

        .alert-right {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .siren-stop {
          padding: 6px 12px;
          background: #dc2626;
          border: none;
          border-radius: 4px;
          color: #fff;
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          font-weight: 600;
          cursor: pointer;
          animation: btnPulse 0.5s infinite;
        }

        @keyframes btnPulse {
          50% { transform: scale(1.05); }
        }

        /* Simulation Controls */
        .sim-dropdown {
          position: relative;
        }

        .sim-btn {
          padding: 6px 12px;
          background: linear-gradient(135deg, #7c3aed, #4f46e5);
          border: 1px solid #8b5cf6;
          border-radius: 4px;
          color: #fff;
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          font-weight: 600;
          cursor: pointer;
        }

        .sim-btn:hover {
          background: linear-gradient(135deg, #8b5cf6, #6366f1);
        }

        .sim-menu {
          position: absolute;
          top: 100%;
          right: 0;
          margin-top: 6px;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 6px;
          min-width: 200px;
          z-index: 1000;
          display: none;
        }

        .sim-dropdown.open .sim-menu {
          display: block;
        }

        .sim-option {
          display: flex;
          align-items: center;
          gap: 10px;
          width: 100%;
          padding: 10px 12px;
          background: transparent;
          border: none;
          border-bottom: 1px solid #262626;
          color: #e5e5e5;
          text-align: left;
          cursor: pointer;
        }

        .sim-option:hover {
          background: #262626;
        }

        .sim-mag {
          font-family: 'JetBrains Mono', monospace;
          font-size: 12px;
          font-weight: 700;
          color: #f87171;
        }

        .sim-name {
          font-size: 11px;
        }

        .sim-cancel {
          width: 100%;
          padding: 8px;
          background: transparent;
          border: none;
          color: #666;
          font-size: 10px;
          cursor: pointer;
        }

        .sim-cancel:hover {
          background: #1f1f1f;
        }

        .sim-active-bar {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 4px 10px;
          background: rgba(220, 38, 38, 0.1);
          border: 1px solid rgba(220, 38, 38, 0.3);
          border-radius: 4px;
        }

        .sim-badge {
          font-family: 'JetBrains Mono', monospace;
          font-size: 9px;
          font-weight: 700;
          color: #fca5a5;
          background: rgba(220, 38, 38, 0.3);
          padding: 2px 6px;
          border-radius: 2px;
          animation: simBlink 1s infinite;
        }

        @keyframes simBlink {
          50% { opacity: 0.5; }
        }

        .sim-time {
          font-family: 'JetBrains Mono', monospace;
          font-size: 12px;
          font-weight: 600;
          color: #fbbf24;
        }

        .sim-wave {
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
          color: #60a5fa;
        }

        .sim-stop {
          padding: 3px 8px;
          background: #dc2626;
          border: none;
          border-radius: 3px;
          color: #fff;
          font-size: 10px;
          cursor: pointer;
        }

        .view-switch {
          padding: 6px 12px;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 4px;
          color: #888;
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          font-weight: 500;
          text-decoration: none;
          transition: all 0.2s;
        }

        .view-switch:hover {
          background: #262626;
          color: #fff;
        }

        /* Main Content */
        .globe-main {
          flex: 1;
          display: flex;
          overflow: hidden;
        }

        .cesium-container {
          flex: 1;
          position: relative;
          height: 100%;
          min-height: 0;
          overflow: hidden;
        }

        /* Sidebar */
        .globe-sidebar {
          width: 300px;
          background: rgba(13, 13, 13, 0.95);
          border-left: 1px solid #222;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        /* Info Cards */
        .info-card {
          margin: 12px;
          background: #141414;
          border: 1px solid #262626;
          border-radius: 6px;
          overflow: hidden;
        }

        .info-card.tsunami {
          border-color: rgba(220, 38, 38, 0.4);
        }

        .card-header {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 10px 12px;
          background: #0a0a0a;
          border-bottom: 1px solid #1f1f1f;
        }

        .card-badge {
          font-family: 'JetBrains Mono', monospace;
          font-size: 9px;
          font-weight: 600;
          color: #10b981;
          background: rgba(16, 185, 129, 0.1);
          padding: 2px 6px;
          border-radius: 2px;
        }

        .card-title {
          flex: 1;
          font-size: 13px;
          font-weight: 600;
          color: #fff;
        }

        .card-mag {
          flex: 1;
          font-family: 'JetBrains Mono', monospace;
          font-size: 16px;
          font-weight: 700;
          color: #ef4444;
        }

        .tsunami-tag {
          font-family: 'JetBrains Mono', monospace;
          font-size: 9px;
          font-weight: 700;
          color: #fff;
          background: #dc2626;
          padding: 2px 6px;
          border-radius: 2px;
        }

        .card-close {
          background: none;
          border: none;
          color: #525252;
          font-size: 18px;
          cursor: pointer;
          padding: 0;
          line-height: 1;
        }

        .card-close:hover {
          color: #a3a3a3;
        }

        .card-body {
          padding: 10px 12px;
        }

        .card-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 6px 0;
          border-bottom: 1px solid #1a1a1a;
        }

        .card-row:last-child {
          border-bottom: none;
        }

        .row-label {
          font-size: 10px;
          color: #525252;
        }

        .row-value {
          font-size: 11px;
          color: #e5e5e5;
        }

        .row-value.status-online { color: #10b981; }
        .row-value.status-offline { color: #ef4444; }

        .amplitude-bar {
          width: 80px;
          height: 5px;
          background: #262626;
          border-radius: 3px;
          overflow: hidden;
        }

        .amplitude-fill {
          height: 100%;
          background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444);
          border-radius: 3px;
        }

        /* Events Panel */
        .events-panel, .stations-panel {
          flex: 1;
          display: flex;
          flex-direction: column;
          min-height: 0;
          border-top: 1px solid #1f1f1f;
        }

        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 10px 12px;
          background: #0a0a0a;
          border-bottom: 1px solid #1f1f1f;
        }

        .panel-title {
          font-size: 10px;
          font-weight: 600;
          color: #666;
          letter-spacing: 1px;
        }

        .panel-count {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          color: #888;
        }

        .events-scroll {
          flex: 1;
          overflow-y: auto;
        }

        .events-scroll::-webkit-scrollbar {
          width: 4px;
        }

        .events-scroll::-webkit-scrollbar-track {
          background: #0d0d0d;
        }

        .events-scroll::-webkit-scrollbar-thumb {
          background: #333;
          border-radius: 2px;
        }

        .event-row {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 8px 12px;
          border-bottom: 1px solid #1a1a1a;
          cursor: pointer;
          transition: background 0.15s;
        }

        .event-row:hover {
          background: #1a1a1a;
        }

        .event-row.tsunami {
          background: rgba(220, 38, 38, 0.1);
        }

        .event-row.io {
          border-left: 2px solid #3b82f6;
        }

        .event-mag {
          font-family: 'JetBrains Mono', monospace;
          font-size: 14px;
          font-weight: 600;
          color: #888;
          min-width: 32px;
        }

        .event-mag.med { color: #f59e0b; }
        .event-mag.high { color: #ef4444; }

        .event-info {
          flex: 1;
          min-width: 0;
        }

        .event-place {
          font-size: 11px;
          color: #e5e5e5;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          display: block;
        }

        .event-meta {
          font-size: 9px;
          color: #525252;
        }

        .io-tag {
          font-family: 'JetBrains Mono', monospace;
          font-size: 8px;
          font-weight: 600;
          color: #3b82f6;
          background: rgba(59, 130, 246, 0.1);
          padding: 2px 4px;
          border-radius: 2px;
        }

        /* Stations Panel */
        .stations-panel {
          flex: none;
          max-height: 120px;
        }

        .stations-grid {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          padding: 10px 12px;
        }

        .station-chip {
          display: flex;
          align-items: center;
          gap: 4px;
          padding: 4px 8px;
          background: #1a1a1a;
          border-radius: 3px;
          cursor: pointer;
          transition: background 0.15s;
        }

        .station-chip:hover {
          background: #262626;
        }

        .station-chip.online .chip-dot {
          background: #10b981;
        }

        .station-chip.offline .chip-dot {
          background: #ef4444;
        }

        .chip-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
        }

        .chip-id {
          font-family: 'JetBrains Mono', monospace;
          font-size: 9px;
          color: #888;
        }
      `}</style>
    </div>
  );
}
