import { useState, useEffect, useRef, useCallback } from 'react';
import Map, { Marker, Popup, NavigationControl } from 'react-map-gl';
import type { MapRef } from 'react-map-gl';
import { motion, AnimatePresence } from 'framer-motion';
import 'mapbox-gl/dist/mapbox-gl.css';
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

// Indian coastal regions for risk assessment
const INDIAN_COASTAL_REGIONS = [
  { name: 'Andaman & Nicobar', lat: 11.7401, lng: 92.6586, population: '0.4M', risk: 'HIGH' },
  { name: 'Tamil Nadu Coast', lat: 10.7905, lng: 79.8428, population: '15M', risk: 'HIGH' },
  { name: 'Andhra Pradesh Coast', lat: 15.9129, lng: 80.7314, population: '8M', risk: 'MEDIUM' },
  { name: 'Odisha Coast', lat: 19.8135, lng: 85.8312, population: '5M', risk: 'MEDIUM' },
  { name: 'West Bengal Coast', lat: 21.9497, lng: 88.0883, population: '4M', risk: 'MEDIUM' },
  { name: 'Kerala Coast', lat: 9.9312, lng: 76.2673, population: '10M', risk: 'HIGH' },
  { name: 'Karnataka Coast', lat: 13.3409, lng: 74.7421, population: '3M', risk: 'MEDIUM' },
  { name: 'Gujarat Coast', lat: 21.1702, lng: 72.8311, population: '6M', risk: 'LOW' },
  { name: 'Maharashtra Coast', lat: 18.9220, lng: 72.8347, population: '12M', risk: 'LOW' },
];

// Other at-risk regions in Indian Ocean
const OTHER_RISK_REGIONS = [
  { name: 'Sri Lanka', risk: 'HIGH', distance: 0 },
  { name: 'Indonesia (Sumatra)', risk: 'CRITICAL', distance: 0 },
  { name: 'Thailand', risk: 'HIGH', distance: 0 },
  { name: 'Myanmar', risk: 'MEDIUM', distance: 0 },
  { name: 'Bangladesh', risk: 'MEDIUM', distance: 0 },
  { name: 'Maldives', risk: 'HIGH', distance: 0 },
  { name: 'Malaysia', risk: 'MEDIUM', distance: 0 },
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

export default function TsunamiMap() {
  const { stations: apiStations, events: liveEvents, indianOceanEvents, connected, loading } = useSeismicStream();

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
  const gainNodesRef = useRef<GainNode[]>([]);
  const simulationIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Emergency Alert Siren - Purge/Civil Defense style
  const playSiren = useCallback(() => {
    if (sirenPlaying) return;

    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioContext;

      // Create multiple oscillators for rich, ominous sound
      const masterGain = audioContext.createGain();
      masterGain.connect(audioContext.destination);
      masterGain.gain.value = 0.4;

      const now = audioContext.currentTime;
      const duration = 30; // 30 seconds

      // Main siren - slow, deep sweep (Purge-style)
      const mainOsc = audioContext.createOscillator();
      const mainGain = audioContext.createGain();
      mainOsc.connect(mainGain);
      mainGain.connect(masterGain);
      mainOsc.type = 'sawtooth';
      mainGain.gain.value = 0.5;

      // Slow, ominous frequency sweep (low to high, then pause)
      for (let i = 0; i < 10; i++) {
        const cycleStart = now + i * 3;
        // Rise slowly from deep bass
        mainOsc.frequency.setValueAtTime(120, cycleStart);
        mainOsc.frequency.linearRampToValueAtTime(480, cycleStart + 2);
        // Hold at peak
        mainOsc.frequency.setValueAtTime(480, cycleStart + 2.5);
        // Quick drop
        mainOsc.frequency.linearRampToValueAtTime(120, cycleStart + 2.8);
      }

      // Sub bass drone for ominous feeling
      const subOsc = audioContext.createOscillator();
      const subGain = audioContext.createGain();
      subOsc.connect(subGain);
      subGain.connect(masterGain);
      subOsc.type = 'sine';
      subOsc.frequency.value = 55; // Deep A note
      subGain.gain.value = 0.3;

      // Pulsing effect on sub bass
      for (let i = 0; i < 30; i++) {
        subGain.gain.setValueAtTime(0.3, now + i);
        subGain.gain.linearRampToValueAtTime(0.1, now + i + 0.5);
        subGain.gain.linearRampToValueAtTime(0.3, now + i + 1);
      }

      // High-frequency alarm overlay
      const alarmOsc = audioContext.createOscillator();
      const alarmGain = audioContext.createGain();
      alarmOsc.connect(alarmGain);
      alarmGain.connect(masterGain);
      alarmOsc.type = 'square';
      alarmGain.gain.value = 0.15;

      // Staccato alarm pattern
      for (let i = 0; i < 60; i++) {
        const t = now + i * 0.5;
        alarmOsc.frequency.setValueAtTime(880, t);
        alarmOsc.frequency.setValueAtTime(660, t + 0.25);
        alarmGain.gain.setValueAtTime(0.15, t);
        alarmGain.gain.setValueAtTime(0, t + 0.2);
        alarmGain.gain.setValueAtTime(0.15, t + 0.25);
        alarmGain.gain.setValueAtTime(0, t + 0.45);
      }

      // Start all oscillators
      mainOsc.start(now);
      subOsc.start(now);
      alarmOsc.start(now);

      mainOsc.stop(now + duration);
      subOsc.stop(now + duration);
      alarmOsc.stop(now + duration);

      oscillatorsRef.current = [mainOsc, subOsc, alarmOsc];
      gainNodesRef.current = [mainGain, subGain, alarmGain, masterGain];

      setSirenPlaying(true);
      setTimeout(() => setSirenPlaying(false), duration * 1000);
    } catch (e) {
      console.error('Audio playback failed:', e);
    }
  }, [sirenPlaying]);

  const stopSiren = useCallback(() => {
    // Stop all oscillators
    oscillatorsRef.current.forEach(osc => {
      try {
        osc.stop();
      } catch (e) {}
    });
    oscillatorsRef.current = [];
    gainNodesRef.current = [];

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
      description: 'Magnitude 9.1 megathrust earthquake - deadliest tsunami in recorded history',
    },
    {
      name: '2011 Japan-like Event',
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
      description: 'Hypothetical megathrust event in the Andaman subduction zone',
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
      description: 'Major earthquake directly threatening Indian east coast',
    },
  ];

  // Start simulation
  const startSimulation = useCallback((scenarioIndex: number = 0) => {
    if (simulationActive) return;

    const scenario = SIMULATION_SCENARIOS[scenarioIndex];
    setSimulationActive(true);
    setSimulatedEvent(scenario.event as APIEvent);
    setWaveRadius(0);
    setSimulationTime(0);
    playSiren();

    // Animate wave propagation (1 second = 10 minutes real time, wave speed ~700km/h)
    const startTime = Date.now();
    simulationIntervalRef.current = setInterval(() => {
      const elapsed = (Date.now() - startTime) / 1000; // seconds since start
      const simulatedMinutes = elapsed * 10; // 1 sec = 10 min
      const waveDistanceKm = (simulatedMinutes / 60) * 700; // km traveled

      setSimulationTime(Math.round(simulatedMinutes));
      setWaveRadius(waveDistanceKm);

      // Stop after 4 hours simulated time (24 seconds real time)
      if (simulatedMinutes > 240) {
        stopSimulation();
      }
    }, 100);
  }, [simulationActive, playSiren]);

  // Stop simulation
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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (simulationIntervalRef.current) {
        clearInterval(simulationIntervalRef.current);
      }
    };
  }, []);

  // Calculate distance from event to India
  const calculateDistanceToIndia = (lat: number, lng: number): number => {
    // Approximate center of Indian east coast
    const indiaLat = 13.0;
    const indiaLng = 80.0;
    const R = 6371; // Earth radius in km
    const dLat = (indiaLat - lat) * Math.PI / 180;
    const dLng = (indiaLng - lng) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat * Math.PI / 180) * Math.cos(indiaLat * Math.PI / 180) *
              Math.sin(dLng/2) * Math.sin(dLng/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  };

  // Assess tsunami risk for India
  useEffect(() => {
    // Include simulated event in assessment
    const allEvents = simulatedEvent ? [...liveEvents, simulatedEvent] : liveEvents;
    const allIndianOceanEvents = simulatedEvent && simulatedEvent.in_region
      ? [...indianOceanEvents, simulatedEvent]
      : indianOceanEvents;

    const tsunamiEvents = allEvents.filter(e => e.tsunami);
    const highMagEvents = allEvents.filter(e => e.magnitude >= 7.0 && e.depth < 100);
    const indianOceanHighMag = allIndianOceanEvents.filter(e => e.magnitude >= 6.5);

    let level: RiskAssessment['level'] = 'OK';
    let indiaStatus = 'NO IMMEDIATE THREAT';
    let affectedRegions: string[] = [];
    let estimatedArrivalTime: string | null = null;
    let threateningEvents: APIEvent[] = [];

    // Check for tsunami warnings
    if (tsunamiEvents.length > 0) {
      const nearestTsunami = tsunamiEvents.reduce((nearest, event) => {
        const dist = calculateDistanceToIndia(event.lat, event.lng);
        return dist < calculateDistanceToIndia(nearest.lat, nearest.lng) ? event : nearest;
      }, tsunamiEvents[0]);

      const distance = calculateDistanceToIndia(nearestTsunami.lat, nearestTsunami.lng);
      const tsunamiSpeed = 700; // km/h approximate
      const arrivalHours = distance / tsunamiSpeed;

      if (distance < 3000) {
        level = 'CRITICAL';
        indiaStatus = 'TSUNAMI WARNING - EVACUATE COASTAL AREAS';
        estimatedArrivalTime = `${Math.round(arrivalHours * 60)} MINUTES`;
        affectedRegions = ['Tamil Nadu Coast', 'Andhra Pradesh Coast', 'Andaman & Nicobar', 'Kerala Coast'];
        threateningEvents = tsunamiEvents;
        playSiren();
      }
    } else if (highMagEvents.length > 0 || indianOceanHighMag.length > 0) {
      const relevantEvents = [...highMagEvents, ...indianOceanHighMag];
      const nearestEvent = relevantEvents.reduce((nearest, event) => {
        const dist = calculateDistanceToIndia(event.lat, event.lng);
        return dist < calculateDistanceToIndia(nearest.lat, nearest.lng) ? event : nearest;
      }, relevantEvents[0]);

      const distance = calculateDistanceToIndia(nearestEvent.lat, nearestEvent.lng);

      if (distance < 2000 && nearestEvent.magnitude >= 7.5) {
        level = 'WARNING';
        indiaStatus = 'HIGH SEISMIC ACTIVITY - MONITOR ALERTS';
        affectedRegions = ['Tamil Nadu Coast', 'Andaman & Nicobar'];
        threateningEvents = relevantEvents;
      } else if (distance < 4000) {
        level = 'WATCH';
        indiaStatus = 'ELEVATED SEISMIC ACTIVITY IN REGION';
        threateningEvents = relevantEvents;
      }
    }

    setRiskAssessment({
      level,
      indiaStatus,
      threateningEvents,
      estimatedArrivalTime,
      affectedRegions,
    });
  }, [liveEvents, indianOceanEvents, simulatedEvent, playSiren]);

  // Update time and station amplitudes
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
      case 'HIGH': return '#ef4444';
      case 'MEDIUM': return '#eab308';
      case 'WATCH': return '#3b82f6';
      case 'LOW': return '#22c55e';
      default: return '#10b981';
    }
  };

  const mapboxToken = 'pk.eyJ1IjoiYW51ZGVlcGFkaSIsImEiOiJjbWlydWsxYTExNDkyM2dvYjU4ZGU3amxmIn0.27yT9p4ulnUqSQpr0VsFWw';

  return (
    <div className="tsunami-system">
      {/* Header Bar */}
      <header className="system-header">
        <div className="header-left">
          <div className="system-title">
            <span className="title-icon">◉</span>
            <span className="title-text">INDIAN OCEAN TSUNAMI EARLY WARNING SYSTEM</span>
          </div>
          <div className="system-subtitle">Real-Time Seismic Monitoring Network</div>
        </div>

        <div className="header-center">
          <div className="utc-clock">
            <span className="clock-label">UTC</span>
            <span className="clock-time">{currentTime.toUTCString().slice(17, 25)}</span>
          </div>
          <div className="clock-date">{currentTime.toISOString().slice(0, 10)}</div>
        </div>

        <div className="header-right">
          <div className={`connection-status ${connected ? 'online' : 'offline'}`}>
            <span className="status-dot" />
            <span className="status-text">{connected ? 'LIVE' : 'OFFLINE'}</span>
          </div>
          <div className="header-stats">
            <div className="stat-item">
              <span className="stat-value">{stations.length}</span>
              <span className="stat-label">STATIONS</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{liveEvents.length}</span>
              <span className="stat-label">EVENTS/24H</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{indianOceanEvents.length}</span>
              <span className="stat-label">INDIAN OCEAN</span>
            </div>
          </div>
        </div>
      </header>

      {/* India Risk Alert Banner */}
      <div className={`india-alert-banner ${riskAssessment.level.toLowerCase()}`}>
        <div className="alert-indicator">
          <span className={`indicator-light ${riskAssessment.level.toLowerCase()}`} />
        </div>
        <div className="alert-content">
          <div className="alert-primary">
            <span className="alert-label">INDIA STATUS:</span>
            <span className={`alert-status ${riskAssessment.level.toLowerCase()}`}>
              {riskAssessment.level === 'OK' ? '✓ OK - NO THREAT' : riskAssessment.indiaStatus}
            </span>
          </div>
          {riskAssessment.estimatedArrivalTime && (
            <div className="alert-eta">
              ETA TO COAST: <strong>{riskAssessment.estimatedArrivalTime}</strong>
            </div>
          )}
          {riskAssessment.affectedRegions.length > 0 && (
            <div className="alert-regions">
              AFFECTED: {riskAssessment.affectedRegions.join(' | ')}
            </div>
          )}
        </div>
        {sirenPlaying && (
          <button className="stop-siren-btn" onClick={stopSiren}>
            STOP ALERT
          </button>
        )}

        {/* View Switch */}
        <a href="/globe" className="view-switch-btn">3D GLOBE</a>

        {/* Simulation Controls */}
        <div className="simulation-controls">
          {!simulationActive ? (
            <div className={`sim-dropdown ${simMenuOpen ? 'open' : ''}`}>
              <button
                className="sim-trigger-btn"
                onClick={() => setSimMenuOpen(!simMenuOpen)}
              >
                ▶ RUN SIMULATION
              </button>
              {simMenuOpen && (
                <div className="sim-menu">
                  <div className="sim-menu-header">SELECT SCENARIO</div>
                  <button onClick={() => { startSimulation(0); setSimMenuOpen(false); }} className="sim-option">
                    <span className="sim-mag">M9.1</span>
                    <div className="sim-details">
                      <span className="sim-name">2004 Sumatra Scenario</span>
                      <span className="sim-desc">Deadliest tsunami in recorded history</span>
                    </div>
                  </button>
                  <button onClick={() => { startSimulation(1); setSimMenuOpen(false); }} className="sim-option">
                    <span className="sim-mag">M8.9</span>
                    <div className="sim-details">
                      <span className="sim-name">Andaman Mega Event</span>
                      <span className="sim-desc">Hypothetical subduction zone rupture</span>
                    </div>
                  </button>
                  <button onClick={() => { startSimulation(2); setSimMenuOpen(false); }} className="sim-option">
                    <span className="sim-mag">M7.8</span>
                    <div className="sim-details">
                      <span className="sim-name">Bay of Bengal Event</span>
                      <span className="sim-desc">Direct threat to Indian east coast</span>
                    </div>
                  </button>
                  <button className="sim-cancel" onClick={() => setSimMenuOpen(false)}>
                    CANCEL
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="sim-active">
              <span className="sim-badge">SIMULATION</span>
              <span className="sim-timer">T+{Math.floor(simulationTime / 60)}h {simulationTime % 60}m</span>
              <span className="sim-wave">Wave: {Math.round(waveRadius)}km</span>
              <button onClick={stopSimulation} className="sim-stop-btn">■ STOP</button>
            </div>
          )}
        </div>
      </div>

      <div className="main-content">
        {/* Left Panel - Risk Assessment */}
        <aside className="left-panel">
          <div className="panel-section">
            <div className="section-header">
              <span className="section-icon">▣</span>
              <span className="section-title">INDIA COASTAL RISK</span>
            </div>
            <div className="risk-grid">
              {INDIAN_COASTAL_REGIONS.map(region => (
                <div
                  key={region.name}
                  className={`risk-item ${riskAssessment.affectedRegions.includes(region.name) ? 'affected' : ''}`}
                >
                  <div className="risk-name">{region.name}</div>
                  <div className="risk-meta">
                    <span className="risk-pop">Pop: {region.population}</span>
                    <span
                      className="risk-level"
                      style={{ color: getRiskColor(region.risk) }}
                    >
                      {region.risk}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel-section">
            <div className="section-header">
              <span className="section-icon">▣</span>
              <span className="section-title">REGIONAL STATUS</span>
            </div>
            <div className="region-list">
              {OTHER_RISK_REGIONS.map(region => (
                <div key={region.name} className="region-item">
                  <span className="region-name">{region.name}</span>
                  <span
                    className="region-risk"
                    style={{ background: getRiskColor(region.risk) }}
                  >
                    {region.risk}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="panel-section">
            <div className="section-header">
              <span className="section-icon">▣</span>
              <span className="section-title">NETWORK METRICS</span>
            </div>
            <div className="metrics-grid">
              <div className="metric">
                <span className="metric-value">{stations.filter(s => s.status === 'online').length}/{stations.length}</span>
                <span className="metric-label">Stations Online</span>
              </div>
              <div className="metric">
                <span className="metric-value">{(stations.reduce((sum, s) => sum + (s.amplitude || 0), 0) / stations.length * 100).toFixed(1)}%</span>
                <span className="metric-label">Avg. Amplitude</span>
              </div>
              <div className="metric">
                <span className="metric-value">{indianOceanEvents.filter(e => e.magnitude >= 5).length}</span>
                <span className="metric-label">M5+ Events (IO)</span>
              </div>
              <div className="metric">
                <span className="metric-value">{liveEvents.filter(e => e.depth < 70).length}</span>
                <span className="metric-label">Shallow Events</span>
              </div>
            </div>
          </div>
        </aside>

        {/* Map */}
        <div className="map-container">
          <Map
            initialViewState={{
              longitude: 85,
              latitude: 5,
              zoom: 3.8,
            }}
            style={{ width: '100%', height: '100%' }}
            mapStyle="mapbox://styles/mapbox/dark-v11"
            mapboxAccessToken={mapboxToken}
          >
            <NavigationControl position="top-right" />

            {/* Station Markers */}
            {stations.map(station => (
              <Marker
                key={station.id}
                longitude={station.lng}
                latitude={station.lat}
                anchor="center"
              >
                <div
                  className={`station-marker ${station.status}`}
                  onClick={() => setSelectedStation(station)}
                >
                  <span className="marker-ring" />
                  <span className="marker-core">{station.id}</span>
                </div>
              </Marker>
            ))}

            {/* Simulated Event & Wave Propagation */}
            {simulatedEvent && (
              <Marker
                longitude={simulatedEvent.lng}
                latitude={simulatedEvent.lat}
                anchor="center"
              >
                <div className="simulated-epicenter">
                  {/* Wave propagation rings */}
                  {waveRadius > 0 && (
                    <>
                      <div
                        className="wave-ring wave-1"
                        style={{
                          width: `${Math.min(waveRadius / 8, 400)}px`,
                          height: `${Math.min(waveRadius / 8, 400)}px`,
                        }}
                      />
                      <div
                        className="wave-ring wave-2"
                        style={{
                          width: `${Math.min(waveRadius / 10, 320)}px`,
                          height: `${Math.min(waveRadius / 10, 320)}px`,
                        }}
                      />
                      <div
                        className="wave-ring wave-3"
                        style={{
                          width: `${Math.min(waveRadius / 12, 250)}px`,
                          height: `${Math.min(waveRadius / 12, 250)}px`,
                        }}
                      />
                    </>
                  )}
                  {/* Epicenter */}
                  <div className="epicenter-core">
                    <span className="epicenter-mag">M{simulatedEvent.magnitude}</span>
                  </div>
                  <div className="epicenter-pulse" />
                </div>
              </Marker>
            )}

            {/* Earthquake Markers with Impact Radius */}
            {liveEvents.slice(0, 20).map(event => {
              const size = Math.max(20, Math.min(50, event.magnitude * 7));
              // Calculate impact radius based on magnitude (rough estimate)
              // M4.5 ~ 50km, M5.5 ~ 150km, M6.5 ~ 450km, M7.5 ~ 1400km
              const impactRadiusKm = Math.pow(10, (event.magnitude - 3) / 1.5) * 15;
              // Convert km to pixels at this zoom level (approximate)
              const impactRadiusPx = Math.min(impactRadiusKm / 3, 300);
              const showImpact = event.magnitude >= 5.0;

              return (
                <Marker
                  key={event.event_id}
                  longitude={event.lng}
                  latitude={event.lat}
                  anchor="center"
                >
                  <div className="earthquake-container">
                    {/* Impact radius zones */}
                    {showImpact && (
                      <>
                        <div
                          className={`impact-zone severe ${event.tsunami ? 'tsunami-zone' : ''}`}
                          style={{
                            width: `${impactRadiusPx * 0.3}px`,
                            height: `${impactRadiusPx * 0.3}px`,
                          }}
                        />
                        <div
                          className={`impact-zone moderate ${event.tsunami ? 'tsunami-zone' : ''}`}
                          style={{
                            width: `${impactRadiusPx * 0.6}px`,
                            height: `${impactRadiusPx * 0.6}px`,
                          }}
                        />
                        <div
                          className={`impact-zone light ${event.tsunami ? 'tsunami-zone' : ''}`}
                          style={{
                            width: `${impactRadiusPx}px`,
                            height: `${impactRadiusPx}px`,
                          }}
                        />
                      </>
                    )}
                    <motion.div
                      className={`earthquake-marker ${event.tsunami ? 'tsunami' : ''} ${event.in_region ? 'in-region' : ''}`}
                      onClick={() => setSelectedEvent(event)}
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      style={{ width: size, height: size }}
                    >
                      <span className="eq-pulse" />
                      <span className="eq-core">{event.magnitude.toFixed(1)}</span>
                    </motion.div>
                    {showImpact && (
                      <div className="impact-label">
                        ~{Math.round(impactRadiusKm)}km radius
                      </div>
                    )}
                  </div>
                </Marker>
              );
            })}

            {/* Popups */}
            {selectedStation && (
              <Popup
                longitude={selectedStation.lng}
                latitude={selectedStation.lat}
                anchor="bottom"
                onClose={() => setSelectedStation(null)}
                closeButton={false}
              >
                <div className="popup-content station">
                  <div className="popup-header">
                    <span className="popup-badge">{selectedStation.network}</span>
                    <span className="popup-title">{selectedStation.id}</span>
                    <button className="popup-close" onClick={() => setSelectedStation(null)}>×</button>
                  </div>
                  <div className="popup-body">
                    <div className="popup-row">
                      <span className="row-label">Location</span>
                      <span className="row-value">{selectedStation.name}, {selectedStation.country}</span>
                    </div>
                    <div className="popup-row">
                      <span className="row-label">Coordinates</span>
                      <span className="row-value">{selectedStation.lat.toFixed(4)}°, {selectedStation.lng.toFixed(4)}°</span>
                    </div>
                    <div className="popup-row">
                      <span className="row-label">Status</span>
                      <span className={`row-value status ${selectedStation.status}`}>{selectedStation.status.toUpperCase()}</span>
                    </div>
                    <div className="popup-row">
                      <span className="row-label">Amplitude</span>
                      <div className="amplitude-bar">
                        <div
                          className="amplitude-fill"
                          style={{
                            width: `${(selectedStation.amplitude || 0) * 100}%`,
                            background: (selectedStation.amplitude || 0) > 0.6 ? '#ef4444' : (selectedStation.amplitude || 0) > 0.3 ? '#f59e0b' : '#10b981'
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </Popup>
            )}

            {selectedEvent && (
              <Popup
                longitude={selectedEvent.lng}
                latitude={selectedEvent.lat}
                anchor="bottom"
                onClose={() => setSelectedEvent(null)}
                closeButton={false}
              >
                <div className={`popup-content event ${selectedEvent.tsunami ? 'tsunami' : ''}`}>
                  <div className="popup-header">
                    <span className="popup-mag">M{selectedEvent.magnitude.toFixed(1)}</span>
                    {selectedEvent.tsunami && <span className="tsunami-tag">TSUNAMI</span>}
                    <button className="popup-close" onClick={() => setSelectedEvent(null)}>×</button>
                  </div>
                  <div className="popup-body">
                    <div className="popup-row">
                      <span className="row-label">Location</span>
                      <span className="row-value">{selectedEvent.place}</span>
                    </div>
                    <div className="popup-row">
                      <span className="row-label">Depth</span>
                      <span className="row-value">{selectedEvent.depth.toFixed(1)} km</span>
                    </div>
                    <div className="popup-row">
                      <span className="row-label">Time</span>
                      <span className="row-value">{new Date(selectedEvent.timestamp).toLocaleString()}</span>
                    </div>
                    <div className="popup-row">
                      <span className="row-label">Distance to India</span>
                      <span className="row-value">{Math.round(calculateDistanceToIndia(selectedEvent.lat, selectedEvent.lng))} km</span>
                    </div>
                  </div>
                </div>
              </Popup>
            )}
          </Map>
        </div>

        {/* Right Panel - Events */}
        <aside className="right-panel">
          <div className="panel-section">
            <div className="section-header">
              <span className="section-icon">▣</span>
              <span className="section-title">RECENT SEISMIC EVENTS</span>
            </div>
            <div className="events-list">
              {liveEvents.slice(0, 12).map(event => (
                <div
                  key={event.event_id}
                  className={`event-item ${event.tsunami ? 'tsunami' : ''} ${event.in_region ? 'in-region' : ''}`}
                  onClick={() => setSelectedEvent(event)}
                >
                  <div className="event-mag-col">
                    <span className={`event-mag ${event.magnitude >= 6 ? 'high' : event.magnitude >= 5 ? 'medium' : ''}`}>
                      {event.magnitude.toFixed(1)}
                    </span>
                    {event.tsunami && <span className="tsunami-indicator">T</span>}
                  </div>
                  <div className="event-details">
                    <div className="event-place">{event.place}</div>
                    <div className="event-meta">
                      <span>{event.depth.toFixed(0)}km depth</span>
                      <span>{new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                    </div>
                  </div>
                  {event.in_region && <span className="io-badge">IO</span>}
                </div>
              ))}
            </div>
          </div>

          <div className="panel-section">
            <div className="section-header">
              <span className="section-icon">▣</span>
              <span className="section-title">THREAT ANALYSIS</span>
            </div>
            <div className="threat-summary">
              {riskAssessment.threateningEvents.length === 0 ? (
                <div className="no-threats">
                  <span className="ok-icon">✓</span>
                  <span>No significant threats detected in the Indian Ocean region.</span>
                </div>
              ) : (
                riskAssessment.threateningEvents.map(event => (
                  <div key={event.event_id} className="threat-item">
                    <div className="threat-header">
                      <span className="threat-mag">M{event.magnitude.toFixed(1)}</span>
                      <span className="threat-place">{event.place}</span>
                    </div>
                    <div className="threat-details">
                      <span>Depth: {event.depth.toFixed(0)}km</span>
                      <span>Distance: {Math.round(calculateDistanceToIndia(event.lat, event.lng))}km</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="panel-section legend-section">
            <div className="section-header">
              <span className="section-icon">▣</span>
              <span className="section-title">LEGEND</span>
            </div>
            <div className="legend-items">
              <div className="legend-item">
                <span className="legend-marker station-legend" />
                <span>Monitoring Station</span>
              </div>
              <div className="legend-item">
                <span className="legend-marker eq-legend" />
                <span>Earthquake</span>
              </div>
              <div className="legend-item">
                <span className="legend-marker tsunami-legend" />
                <span>Tsunami Warning</span>
              </div>
              <div className="legend-item">
                <span className="legend-marker io-legend" />
                <span>Indian Ocean Event</span>
              </div>
            </div>
          </div>
        </aside>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        * {
          box-sizing: border-box;
        }

        .tsunami-system {
          position: relative;
          width: 100%;
          height: 100vh;
          background: #0a0a0a;
          font-family: 'IBM Plex Sans', -apple-system, sans-serif;
          color: #e5e5e5;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        /* Header */
        .system-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px 20px;
          background: linear-gradient(180deg, #141414 0%, #0d0d0d 100%);
          border-bottom: 1px solid #262626;
          z-index: 100;
        }

        .header-left {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .system-title {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .title-icon {
          color: #10b981;
          font-size: 12px;
        }

        .title-text {
          font-size: 14px;
          font-weight: 600;
          letter-spacing: 1px;
          color: #f5f5f5;
        }

        .system-subtitle {
          font-size: 11px;
          color: #737373;
          padding-left: 22px;
        }

        .header-center {
          display: flex;
          flex-direction: column;
          align-items: center;
        }

        .utc-clock {
          display: flex;
          align-items: baseline;
          gap: 8px;
        }

        .clock-label {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 10px;
          color: #525252;
        }

        .clock-time {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 24px;
          font-weight: 600;
          color: #f5f5f5;
          letter-spacing: 2px;
        }

        .clock-date {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 11px;
          color: #525252;
        }

        .header-right {
          display: flex;
          align-items: center;
          gap: 24px;
        }

        .connection-status {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 6px 12px;
          border-radius: 4px;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 11px;
          font-weight: 600;
        }

        .connection-status.online {
          background: rgba(16, 185, 129, 0.1);
          border: 1px solid rgba(16, 185, 129, 0.3);
          color: #10b981;
        }

        .connection-status.offline {
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          color: #ef4444;
        }

        .status-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: currentColor;
        }

        .connection-status.online .status-dot {
          animation: blink 2s infinite;
        }

        @keyframes blink {
          0%, 50%, 100% { opacity: 1; }
          25%, 75% { opacity: 0.4; }
        }

        .header-stats {
          display: flex;
          gap: 20px;
        }

        .stat-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 2px;
        }

        .stat-value {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 18px;
          font-weight: 600;
          color: #f5f5f5;
        }

        .stat-label {
          font-size: 9px;
          color: #525252;
          letter-spacing: 1px;
        }

        /* India Alert Banner */
        .india-alert-banner {
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 10px 20px;
          border-bottom: 1px solid #262626;
        }

        .india-alert-banner.ok {
          background: linear-gradient(90deg, rgba(16, 185, 129, 0.05) 0%, transparent 50%);
        }

        .india-alert-banner.watch {
          background: linear-gradient(90deg, rgba(59, 130, 246, 0.1) 0%, transparent 50%);
        }

        .india-alert-banner.warning {
          background: linear-gradient(90deg, rgba(245, 158, 11, 0.1) 0%, transparent 50%);
        }

        .india-alert-banner.critical {
          background: linear-gradient(90deg, rgba(220, 38, 38, 0.15) 0%, rgba(220, 38, 38, 0.05) 50%, transparent 100%);
          animation: criticalPulse 1s infinite;
        }

        @keyframes criticalPulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.8; }
        }

        .alert-indicator {
          width: 40px;
          display: flex;
          justify-content: center;
        }

        .indicator-light {
          width: 12px;
          height: 12px;
          border-radius: 50%;
        }

        .indicator-light.ok { background: #10b981; }
        .indicator-light.watch { background: #3b82f6; }
        .indicator-light.warning { background: #f59e0b; animation: pulse 1s infinite; }
        .indicator-light.critical { background: #dc2626; animation: pulse 0.5s infinite; }

        @keyframes pulse {
          0%, 100% { box-shadow: 0 0 0 0 currentColor; }
          50% { box-shadow: 0 0 0 8px transparent; }
        }

        .alert-content {
          flex: 1;
          display: flex;
          align-items: center;
          gap: 24px;
        }

        .alert-primary {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .alert-label {
          font-size: 12px;
          font-weight: 600;
          color: #737373;
        }

        .alert-status {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 13px;
          font-weight: 600;
        }

        .alert-status.ok { color: #10b981; }
        .alert-status.watch { color: #3b82f6; }
        .alert-status.warning { color: #f59e0b; }
        .alert-status.critical { color: #dc2626; }

        .alert-eta {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 12px;
          color: #dc2626;
        }

        .alert-eta strong {
          color: #fca5a5;
        }

        .alert-regions {
          font-size: 11px;
          color: #a3a3a3;
        }

        .stop-siren-btn {
          padding: 8px 16px;
          background: #dc2626;
          border: none;
          border-radius: 4px;
          color: white;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 11px;
          font-weight: 600;
          cursor: pointer;
          animation: btnPulse 0.5s infinite;
        }

        @keyframes btnPulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.05); }
        }

        /* Simulation Controls */
        .simulation-controls {
          margin-left: auto;
        }

        .sim-dropdown {
          position: relative;
        }

        .sim-trigger-btn {
          padding: 8px 16px;
          background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
          border: 1px solid #8b5cf6;
          border-radius: 4px;
          color: white;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 11px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }

        .sim-trigger-btn:hover {
          background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
          transform: translateY(-1px);
        }

        .sim-menu {
          position: absolute;
          top: 100%;
          right: 0;
          margin-top: 8px;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 6px;
          overflow: hidden;
          display: none;
          min-width: 240px;
          z-index: 1000;
          box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }

        .sim-dropdown.open .sim-menu {
          display: block;
        }

        .sim-option {
          display: flex;
          align-items: center;
          gap: 12px;
          width: 100%;
          padding: 12px 16px;
          background: transparent;
          border: none;
          border-bottom: 1px solid #262626;
          color: #e5e5e5;
          text-align: left;
          cursor: pointer;
          transition: background 0.2s;
        }

        .sim-option:last-child {
          border-bottom: none;
        }

        .sim-option:hover {
          background: #262626;
        }

        .sim-mag {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 14px;
          font-weight: 700;
          color: #f87171;
          min-width: 40px;
        }

        .sim-name {
          font-size: 12px;
          color: #e5e5e5;
          font-weight: 500;
        }

        .sim-menu-header {
          padding: 12px 16px;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 10px;
          font-weight: 700;
          color: #737373;
          letter-spacing: 1px;
          background: #141414;
          border-bottom: 1px solid #262626;
        }

        .sim-details {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .sim-desc {
          font-size: 10px;
          color: #737373;
        }

        .sim-cancel {
          width: 100%;
          padding: 10px 16px;
          background: #1a1a1a;
          border: none;
          border-top: 1px solid #262626;
          color: #737373;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 10px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }

        .sim-cancel:hover {
          background: #262626;
          color: #a3a3a3;
        }

        .view-switch-btn {
          padding: 8px 16px;
          background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%);
          border: 1px solid #22d3ee;
          border-radius: 4px;
          color: white;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 11px;
          font-weight: 600;
          cursor: pointer;
          text-decoration: none;
          transition: all 0.2s;
          margin-right: 12px;
        }

        .view-switch-btn:hover {
          background: linear-gradient(135deg, #06b6d4 0%, #22d3ee 100%);
          transform: translateY(-1px);
        }

        .sim-active {
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 6px 12px;
          background: rgba(220, 38, 38, 0.1);
          border: 1px solid rgba(220, 38, 38, 0.3);
          border-radius: 4px;
        }

        .sim-badge {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 10px;
          font-weight: 700;
          color: #fca5a5;
          background: rgba(220, 38, 38, 0.3);
          padding: 2px 8px;
          border-radius: 2px;
          animation: simBlink 1s infinite;
        }

        @keyframes simBlink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        .sim-timer {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 14px;
          font-weight: 600;
          color: #fbbf24;
        }

        .sim-wave {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 12px;
          color: #60a5fa;
        }

        .sim-stop-btn {
          padding: 4px 12px;
          background: #dc2626;
          border: none;
          border-radius: 3px;
          color: white;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 10px;
          font-weight: 600;
          cursor: pointer;
        }

        .sim-stop-btn:hover {
          background: #ef4444;
        }

        /* Simulated Epicenter & Wave */
        .simulated-epicenter {
          position: relative;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .wave-ring {
          position: absolute;
          border: 2px solid rgba(220, 38, 38, 0.6);
          border-radius: 50%;
          animation: waveExpand 2s infinite ease-out;
          pointer-events: none;
        }

        .wave-ring.wave-1 {
          animation-delay: 0s;
          border-color: rgba(220, 38, 38, 0.8);
        }

        .wave-ring.wave-2 {
          animation-delay: 0.5s;
          border-color: rgba(220, 38, 38, 0.5);
        }

        .wave-ring.wave-3 {
          animation-delay: 1s;
          border-color: rgba(220, 38, 38, 0.3);
        }

        @keyframes waveExpand {
          0% { opacity: 1; transform: scale(0.8); }
          100% { opacity: 0; transform: scale(1.5); }
        }

        .epicenter-core {
          width: 60px;
          height: 60px;
          background: radial-gradient(circle, #dc2626 0%, #991b1b 70%, #450a0a 100%);
          border: 3px solid #fca5a5;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 10;
          box-shadow: 0 0 30px rgba(220, 38, 38, 0.8), 0 0 60px rgba(220, 38, 38, 0.4);
        }

        .epicenter-mag {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 14px;
          font-weight: 700;
          color: white;
          text-shadow: 0 0 10px rgba(0,0,0,0.5);
        }

        .epicenter-pulse {
          position: absolute;
          width: 80px;
          height: 80px;
          border: 2px solid #fca5a5;
          border-radius: 50%;
          animation: epicenterPulse 1s infinite;
        }

        @keyframes epicenterPulse {
          0% { transform: scale(1); opacity: 1; }
          100% { transform: scale(2); opacity: 0; }
        }

        /* Earthquake Impact Zones */
        .earthquake-container {
          position: relative;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .impact-zone {
          position: absolute;
          border-radius: 50%;
          pointer-events: none;
          opacity: 0.3;
        }

        .impact-zone.severe {
          background: radial-gradient(circle, rgba(220, 38, 38, 0.4) 0%, rgba(220, 38, 38, 0) 70%);
          border: 1px dashed rgba(220, 38, 38, 0.5);
        }

        .impact-zone.moderate {
          background: radial-gradient(circle, rgba(251, 146, 60, 0.3) 0%, rgba(251, 146, 60, 0) 70%);
          border: 1px dashed rgba(251, 146, 60, 0.4);
        }

        .impact-zone.light {
          background: radial-gradient(circle, rgba(250, 204, 21, 0.2) 0%, rgba(250, 204, 21, 0) 70%);
          border: 1px dashed rgba(250, 204, 21, 0.3);
        }

        .impact-zone.tsunami-zone {
          animation: tsunamiPulse 2s infinite ease-out;
        }

        @keyframes tsunamiPulse {
          0% { opacity: 0.3; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(1.05); }
          100% { opacity: 0.3; transform: scale(1); }
        }

        .impact-label {
          position: absolute;
          bottom: -20px;
          left: 50%;
          transform: translateX(-50%);
          font-family: 'IBM Plex Mono', monospace;
          font-size: 9px;
          color: rgba(250, 204, 21, 0.8);
          white-space: nowrap;
          background: rgba(0, 0, 0, 0.6);
          padding: 2px 6px;
          border-radius: 3px;
        }

        /* Main Content */
        .main-content {
          flex: 1;
          display: flex;
          overflow: hidden;
        }

        /* Side Panels */
        .left-panel, .right-panel {
          width: 280px;
          background: #0d0d0d;
          border-right: 1px solid #1f1f1f;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
        }

        .right-panel {
          border-right: none;
          border-left: 1px solid #1f1f1f;
        }

        .panel-section {
          padding: 16px;
          border-bottom: 1px solid #1f1f1f;
        }

        .section-header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 12px;
        }

        .section-icon {
          color: #525252;
          font-size: 10px;
        }

        .section-title {
          font-size: 10px;
          font-weight: 600;
          color: #737373;
          letter-spacing: 1px;
        }

        /* Risk Grid */
        .risk-grid {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .risk-item {
          padding: 10px 12px;
          background: #141414;
          border: 1px solid #1f1f1f;
          border-radius: 4px;
          transition: all 0.2s;
        }

        .risk-item.affected {
          background: rgba(220, 38, 38, 0.1);
          border-color: rgba(220, 38, 38, 0.3);
        }

        .risk-name {
          font-size: 12px;
          font-weight: 500;
          color: #e5e5e5;
          margin-bottom: 4px;
        }

        .risk-meta {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .risk-pop {
          font-size: 10px;
          color: #525252;
        }

        .risk-level {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 10px;
          font-weight: 600;
        }

        /* Region List */
        .region-list {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .region-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 10px;
          background: #141414;
          border-radius: 4px;
        }

        .region-name {
          font-size: 12px;
          color: #a3a3a3;
        }

        .region-risk {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 9px;
          font-weight: 600;
          padding: 2px 6px;
          border-radius: 2px;
          color: white;
        }

        /* Metrics */
        .metrics-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
        }

        .metric {
          padding: 12px;
          background: #141414;
          border-radius: 4px;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 4px;
        }

        .metric-value {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 18px;
          font-weight: 600;
          color: #f5f5f5;
        }

        .metric-label {
          font-size: 9px;
          color: #525252;
          text-align: center;
        }

        /* Events List */
        .events-list {
          display: flex;
          flex-direction: column;
          gap: 4px;
          max-height: 350px;
          overflow-y: auto;
        }

        .events-list::-webkit-scrollbar {
          width: 4px;
        }

        .events-list::-webkit-scrollbar-track {
          background: #141414;
        }

        .events-list::-webkit-scrollbar-thumb {
          background: #333;
          border-radius: 2px;
        }

        .event-item {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 10px;
          background: #141414;
          border: 1px solid transparent;
          border-radius: 4px;
          cursor: pointer;
          transition: all 0.15s;
        }

        .event-item:hover {
          background: #1a1a1a;
          border-color: #333;
        }

        .event-item.tsunami {
          background: rgba(220, 38, 38, 0.1);
          border-color: rgba(220, 38, 38, 0.3);
        }

        .event-item.in-region {
          border-left: 2px solid #3b82f6;
        }

        .event-mag-col {
          display: flex;
          flex-direction: column;
          align-items: center;
          min-width: 36px;
        }

        .event-mag {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 16px;
          font-weight: 600;
          color: #a3a3a3;
        }

        .event-mag.medium { color: #f59e0b; }
        .event-mag.high { color: #ef4444; }

        .tsunami-indicator {
          font-size: 9px;
          font-weight: 700;
          color: #dc2626;
          margin-top: 2px;
        }

        .event-details {
          flex: 1;
          min-width: 0;
        }

        .event-place {
          font-size: 11px;
          color: #e5e5e5;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          margin-bottom: 3px;
        }

        .event-meta {
          display: flex;
          gap: 10px;
          font-size: 10px;
          color: #525252;
        }

        .io-badge {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 9px;
          font-weight: 600;
          color: #3b82f6;
          background: rgba(59, 130, 246, 0.1);
          padding: 2px 5px;
          border-radius: 2px;
        }

        /* Threat Analysis */
        .threat-summary {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .no-threats {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 16px;
          background: rgba(16, 185, 129, 0.05);
          border: 1px solid rgba(16, 185, 129, 0.2);
          border-radius: 4px;
          font-size: 12px;
          color: #10b981;
        }

        .ok-icon {
          font-size: 16px;
        }

        .threat-item {
          padding: 12px;
          background: rgba(220, 38, 38, 0.05);
          border: 1px solid rgba(220, 38, 38, 0.2);
          border-radius: 4px;
        }

        .threat-header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 6px;
        }

        .threat-mag {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 14px;
          font-weight: 600;
          color: #ef4444;
        }

        .threat-place {
          font-size: 12px;
          color: #e5e5e5;
        }

        .threat-details {
          display: flex;
          gap: 16px;
          font-size: 11px;
          color: #737373;
        }

        /* Legend */
        .legend-section {
          margin-top: auto;
        }

        .legend-items {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: 10px;
          font-size: 11px;
          color: #737373;
        }

        .legend-marker {
          width: 12px;
          height: 12px;
          border-radius: 50%;
        }

        .station-legend {
          background: #10b981;
          box-shadow: 0 0 6px rgba(16, 185, 129, 0.5);
        }

        .eq-legend {
          background: #ef4444;
        }

        .tsunami-legend {
          background: #dc2626;
          box-shadow: 0 0 6px rgba(220, 38, 38, 0.5);
        }

        .io-legend {
          background: #3b82f6;
          border: 2px solid #60a5fa;
        }

        /* Map Container */
        .map-container {
          flex: 1;
          position: relative;
        }

        /* Map Markers */
        .station-marker {
          width: 32px;
          height: 32px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          position: relative;
          background: rgba(16, 185, 129, 0.2);
          border: 2px solid #10b981;
        }

        .marker-ring {
          position: absolute;
          inset: -3px;
          border-radius: 50%;
          border: 1px solid rgba(16, 185, 129, 0.5);
          animation: markerPulse 2s infinite;
        }

        @keyframes markerPulse {
          0% { transform: scale(1); opacity: 0.8; }
          100% { transform: scale(1.4); opacity: 0; }
        }

        .marker-core {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 8px;
          font-weight: 600;
          color: #10b981;
        }

        .earthquake-marker {
          position: relative;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
        }

        .eq-pulse {
          position: absolute;
          inset: 0;
          border-radius: 50%;
          background: rgba(239, 68, 68, 0.3);
          animation: eqPulse 1.5s infinite;
        }

        @keyframes eqPulse {
          0% { transform: scale(1); opacity: 0.6; }
          100% { transform: scale(2); opacity: 0; }
        }

        .earthquake-marker.tsunami .eq-pulse {
          background: rgba(220, 38, 38, 0.5);
        }

        .eq-core {
          position: relative;
          width: 70%;
          height: 70%;
          border-radius: 50%;
          background: #ef4444;
          border: 2px solid #fca5a5;
          display: flex;
          align-items: center;
          justify-content: center;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 9px;
          font-weight: 700;
          color: white;
        }

        .earthquake-marker.tsunami .eq-core {
          background: #dc2626;
          box-shadow: 0 0 15px rgba(220, 38, 38, 0.6);
        }

        .earthquake-marker.in-region .eq-core {
          border-color: #60a5fa;
        }

        /* Popups */
        .mapboxgl-popup-content {
          background: transparent !important;
          padding: 0 !important;
          box-shadow: none !important;
        }

        .mapboxgl-popup-tip {
          display: none !important;
        }

        .popup-content {
          background: #141414;
          border: 1px solid #333;
          border-radius: 6px;
          min-width: 220px;
          overflow: hidden;
        }

        .popup-content.tsunami {
          border-color: rgba(220, 38, 38, 0.5);
        }

        .popup-header {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 10px 12px;
          background: #0d0d0d;
          border-bottom: 1px solid #262626;
        }

        .popup-badge {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 9px;
          font-weight: 600;
          color: #10b981;
          background: rgba(16, 185, 129, 0.1);
          padding: 2px 6px;
          border-radius: 2px;
        }

        .popup-title {
          font-size: 14px;
          font-weight: 600;
          color: #f5f5f5;
          flex: 1;
        }

        .popup-mag {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 18px;
          font-weight: 700;
          color: #ef4444;
          flex: 1;
        }

        .tsunami-tag {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 9px;
          font-weight: 700;
          color: white;
          background: #dc2626;
          padding: 3px 6px;
          border-radius: 2px;
        }

        .popup-close {
          background: none;
          border: none;
          color: #525252;
          font-size: 18px;
          cursor: pointer;
          padding: 0;
          line-height: 1;
        }

        .popup-close:hover {
          color: #a3a3a3;
        }

        .popup-body {
          padding: 12px;
        }

        .popup-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 6px 0;
          border-bottom: 1px solid #1f1f1f;
        }

        .popup-row:last-child {
          border-bottom: none;
        }

        .row-label {
          font-size: 11px;
          color: #525252;
        }

        .row-value {
          font-size: 12px;
          color: #e5e5e5;
        }

        .row-value.status.online { color: #10b981; }
        .row-value.status.offline { color: #ef4444; }
        .row-value.status.delayed { color: #f59e0b; }

        .amplitude-bar {
          width: 100px;
          height: 6px;
          background: #262626;
          border-radius: 3px;
          overflow: hidden;
        }

        .amplitude-fill {
          height: 100%;
          border-radius: 3px;
          transition: width 0.3s;
        }

        /* Mapbox Overrides */
        .mapboxgl-ctrl-group {
          background: #141414 !important;
          border: 1px solid #333 !important;
        }

        .mapboxgl-ctrl-group button {
          background: transparent !important;
        }

        .mapboxgl-ctrl-icon {
          filter: invert(0.8);
        }

        .mapboxgl-ctrl-attrib {
          background: rgba(0, 0, 0, 0.6) !important;
          font-size: 10px;
        }

        .mapboxgl-ctrl-attrib a {
          color: #525252 !important;
        }
      `}</style>
    </div>
  );
}
