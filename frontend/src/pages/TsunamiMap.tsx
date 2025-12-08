import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import Map, { Marker, Popup, NavigationControl, Source, Layer } from 'react-map-gl';
import { motion } from 'framer-motion';
import 'mapbox-gl/dist/mapbox-gl.css';
import { useSeismicStream, SeismicEvent as APIEvent } from '../hooks/useSeismicStream';
import circle from '@turf/circle';

// Global monitoring stations
const STATIONS = [
  // Pacific Ring of Fire
  { id: 'MAJO', network: 'IU', name: 'Matsushiro', country: 'Japan', lat: 36.5457, lng: 138.2041, status: 'online' },
  { id: 'TATO', network: 'IU', name: 'Taipei', country: 'Taiwan', lat: 24.9735, lng: 121.4971, status: 'online' },
  { id: 'GUMO', network: 'IU', name: 'Guam', country: 'USA', lat: 13.5893, lng: 144.8684, status: 'online' },
  { id: 'POHA', network: 'IU', name: 'Pohakuloa', country: 'Hawaii', lat: 19.7573, lng: -155.5326, status: 'online' },
  { id: 'ANMO', network: 'IU', name: 'Albuquerque', country: 'USA', lat: 34.9459, lng: -106.4572, status: 'online' },
  // Indian Ocean
  { id: 'PALK', network: 'II', name: 'Pallekele', country: 'Sri Lanka', lat: 7.2728, lng: 80.7022, status: 'online' },
  { id: 'COCO', network: 'II', name: 'Cocos Islands', country: 'Australia', lat: -12.1901, lng: 96.8349, status: 'online' },
  { id: 'DGAR', network: 'II', name: 'Diego Garcia', country: 'BIOT', lat: -7.4121, lng: 72.4525, status: 'online' },
  // Atlantic & Caribbean
  { id: 'SJG', network: 'IU', name: 'San Juan', country: 'Puerto Rico', lat: 18.1091, lng: -66.1500, status: 'online' },
  { id: 'TEIG', network: 'IU', name: 'Tenerife', country: 'Spain', lat: 28.4801, lng: -16.3113, status: 'online' },
  // South America
  { id: 'LVC', network: 'II', name: 'Limon Verde', country: 'Chile', lat: -22.6127, lng: -68.9111, status: 'online' },
  // Europe & Mediterranean
  { id: 'GRFO', network: 'II', name: 'Grafenberg', country: 'Germany', lat: 49.6909, lng: 11.2203, status: 'online' },
];

// Global coastal risk zones
const GLOBAL_COASTAL_REGIONS = [
  { name: 'Japan Pacific Coast', lat: 35.6762, lng: 139.6503, population: '40M', risk: 'HIGH' },
  { name: 'Indonesia', lat: -6.2088, lng: 106.8456, population: '150M', risk: 'CRITICAL' },
  { name: 'Chile Coast', lat: -33.4489, lng: -70.6693, population: '10M', risk: 'HIGH' },
  { name: 'US West Coast', lat: 34.0522, lng: -118.2437, population: '25M', risk: 'MEDIUM' },
  { name: 'Hawaii', lat: 21.3069, lng: -157.8583, population: '1.4M', risk: 'HIGH' },
  { name: 'Philippines', lat: 14.5995, lng: 120.9842, population: '30M', risk: 'HIGH' },
  { name: 'New Zealand', lat: -41.2865, lng: 174.7762, population: '2M', risk: 'MEDIUM' },
  { name: 'Mediterranean', lat: 36.8969, lng: 14.5146, population: '50M', risk: 'LOW' },
  { name: 'Caribbean', lat: 18.2208, lng: -66.5901, population: '20M', risk: 'MEDIUM' },
];

// Global tsunami-prone regions by basin
const TSUNAMI_BASINS = [
  { name: 'Pacific Ring of Fire', risk: 'CRITICAL', distance: 0 },
  { name: 'Indian Ocean', risk: 'HIGH', distance: 0 },
  { name: 'Mediterranean Sea', risk: 'MEDIUM', distance: 0 },
  { name: 'Caribbean Sea', risk: 'MEDIUM', distance: 0 },
  { name: 'Atlantic Ocean', risk: 'LOW', distance: 0 },
  { name: 'Cascadia Zone', risk: 'HIGH', distance: 0 },
  { name: 'Alaska-Aleutian', risk: 'HIGH', distance: 0 },
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
  globalStatus: string;
  threateningEvents: APIEvent[];
  tsunamiCount: number;
  majorEventCount: number;
}

export default function TsunamiMap() {
  const { stations: apiStations, events: liveEvents, indianOceanEvents, connected, loading } = useSeismicStream();

  const [stations, setStations] = useState<Station[]>(STATIONS.map(s => ({ ...s, amplitude: 0.2 })));
  const [selectedStation, setSelectedStation] = useState<Station | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<APIEvent | null>(null);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [riskAssessment, setRiskAssessment] = useState<RiskAssessment>({
    level: 'OK',
    globalStatus: 'NO IMMEDIATE THREAT',
    threateningEvents: [],
    tsunamiCount: 0,
    majorEventCount: 0,
  });
  const [sirenPlaying, setSirenPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [simulationActive, setSimulationActive] = useState(false);
  const [simulatedEvent, setSimulatedEvent] = useState<APIEvent | null>(null);
  const [waveRadius, setWaveRadius] = useState(0);
  const [simulationTime, setSimulationTime] = useState(0);
  const [simMenuOpen, setSimMenuOpen] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const simulationIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Emergency Alert Siren using MP3 audio file
  const playSiren = useCallback(() => {
    if (sirenPlaying || muted) return;

    try {
      // Create or reuse audio element
      if (!audioRef.current) {
        audioRef.current = new Audio('/emergency-alert.mp3');
        audioRef.current.loop = true;
      }

      audioRef.current.currentTime = 0;
      audioRef.current.play().then(() => {
        setSirenPlaying(true);
      }).catch((e) => {
        console.error('Audio playback failed:', e);
      });
    } catch (e) {
      console.error('Audio playback failed:', e);
    }
  }, [sirenPlaying, muted]);

  const stopSiren = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setSirenPlaying(false);
  }, []);

  // Toggle mute
  const toggleMute = useCallback(() => {
    setMuted(prev => {
      const newMuted = !prev;
      if (newMuted && audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
        setSirenPlaying(false);
      }
      return newMuted;
    });
  }, []);

  // Generate GeoJSON for simulation wave propagation circles (geographic, zoom-independent)
  const waveCirclesGeoJSON = useMemo(() => {
    if (!simulatedEvent || waveRadius <= 0) {
      return { type: 'FeatureCollection' as const, features: [] };
    }

    const numRings = 5;
    const features = [];

    for (let i = 0; i < numRings; i++) {
      const ringRadiusKm = waveRadius * (1 - i * 0.15);
      if (ringRadiusKm <= 0) continue;

      const circleFeature = circle(
        [simulatedEvent.lng, simulatedEvent.lat],
        ringRadiusKm,
        { steps: 64, units: 'kilometers' }
      );
      circleFeature.properties = { ring: i, opacity: 0.6 - i * 0.1 };
      features.push(circleFeature);
    }

    return { type: 'FeatureCollection' as const, features };
  }, [simulatedEvent, waveRadius]);

  // Generate GeoJSON for tsunami warning impact zones (only for tsunami events)
  const tsunamiImpactGeoJSON = useMemo(() => {
    // Filter only tsunami events (including simulated)
    const allEvents = simulatedEvent ? [...liveEvents, simulatedEvent] : liveEvents;
    const tsunamiEvents = allEvents.filter(e => e.tsunami);

    if (tsunamiEvents.length === 0) {
      return { type: 'FeatureCollection' as const, features: [] };
    }

    const features: any[] = [];

    tsunamiEvents.forEach(event => {
      // Calculate impact radius based on magnitude (larger for bigger quakes)
      const baseRadiusKm = Math.pow(10, (event.magnitude - 3) / 1.5) * 15;

      // Severe zone (inner) - strongest shaking/immediate danger
      const severeCircle = circle(
        [event.lng, event.lat],
        baseRadiusKm * 0.3,
        { steps: 64, units: 'kilometers' }
      );
      severeCircle.properties = {
        zoneType: 'severe',
        opacity: 0.25,
        eventId: event.event_id
      };
      features.push(severeCircle);

      // Moderate zone (middle)
      const moderateCircle = circle(
        [event.lng, event.lat],
        baseRadiusKm * 0.6,
        { steps: 64, units: 'kilometers' }
      );
      moderateCircle.properties = {
        zoneType: 'moderate',
        opacity: 0.15,
        eventId: event.event_id
      };
      features.push(moderateCircle);

      // Light zone (outer) - tsunami wave reach
      const lightCircle = circle(
        [event.lng, event.lat],
        baseRadiusKm,
        { steps: 64, units: 'kilometers' }
      );
      lightCircle.properties = {
        zoneType: 'light',
        opacity: 0.1,
        eventId: event.event_id
      };
      features.push(lightCircle);
    });

    return { type: 'FeatureCollection' as const, features };
  }, [liveEvents, simulatedEvent]);

  // Simulation scenarios - Global
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
      name: '2011 Japan Earthquake',
      event: {
        event_id: 'sim_japan_2011',
        timestamp: new Date().toISOString(),
        lat: 38.322,
        lng: 142.369,
        magnitude: 9.1,
        depth: 29,
        place: 'Near the east coast of Honshu, Japan',
        tsunami: true,
        in_region: true,
      },
      description: 'Tōhoku earthquake - triggered Fukushima nuclear disaster',
    },
    {
      name: 'Cascadia Subduction Zone',
      event: {
        event_id: 'sim_cascadia',
        timestamp: new Date().toISOString(),
        lat: 44.5,
        lng: -125.0,
        magnitude: 9.0,
        depth: 20,
        place: 'Cascadia Subduction Zone, Pacific Northwest',
        tsunami: true,
        in_region: true,
      },
      description: 'Hypothetical megathrust threatening US/Canada Pacific coast',
    },
    {
      name: '1960 Chile Earthquake',
      event: {
        event_id: 'sim_chile_1960',
        timestamp: new Date().toISOString(),
        lat: -38.29,
        lng: -73.05,
        magnitude: 9.5,
        depth: 25,
        place: 'Valdivia, Chile',
        tsunami: true,
        in_region: true,
      },
      description: 'Largest recorded earthquake in history - trans-Pacific tsunami',
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

  // Assess global tsunami risk
  useEffect(() => {
    // Include simulated event in assessment
    const allEvents = simulatedEvent ? [...liveEvents, simulatedEvent] : liveEvents;

    const tsunamiEvents = allEvents.filter(e => e.tsunami);
    const highMagEvents = allEvents.filter(e => e.magnitude >= 7.0 && e.depth < 100);
    const majorEvents = allEvents.filter(e => e.magnitude >= 6.0);

    let level: RiskAssessment['level'] = 'OK';
    let globalStatus = 'NO IMMEDIATE THREAT';
    let threateningEvents: APIEvent[] = [];

    // Check for tsunami warnings
    if (tsunamiEvents.length > 0) {
      level = 'CRITICAL';
      globalStatus = `TSUNAMI WARNING - ${tsunamiEvents.length} ACTIVE`;
      threateningEvents = tsunamiEvents;
      playSiren();
    } else if (highMagEvents.length > 0) {
      level = 'WARNING';
      globalStatus = `HIGH SEISMIC ACTIVITY - ${highMagEvents.length} MAJOR EVENT${highMagEvents.length > 1 ? 'S' : ''}`;
      threateningEvents = highMagEvents;
    } else if (majorEvents.length > 0) {
      level = 'WATCH';
      globalStatus = `ELEVATED ACTIVITY - ${majorEvents.length} SIGNIFICANT EVENT${majorEvents.length > 1 ? 'S' : ''}`;
      threateningEvents = majorEvents;
    }

    setRiskAssessment({
      level,
      globalStatus,
      threateningEvents,
      tsunamiCount: tsunamiEvents.length,
      majorEventCount: majorEvents.length,
    });
  }, [liveEvents, simulatedEvent, playSiren]);

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
            <span className="title-text">GLOBAL TSUNAMI WARNING SYSTEM</span>
          </div>
          <div className="system-subtitle">Real-Time Seismic Monitoring Network | 2D MAP VIEW</div>
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
              <span className="stat-value">{liveEvents.filter(e => e.magnitude >= 5).length}</span>
              <span className="stat-label">M5+ EVENTS</span>
            </div>
          </div>
        </div>
      </header>

      {/* Global Alert Banner */}
      <div className={`india-alert-banner ${riskAssessment.level.toLowerCase()}`}>
        <div className="alert-indicator">
          <span className={`indicator-light ${riskAssessment.level.toLowerCase()}`} />
        </div>
        <div className="alert-content">
          <div className="alert-primary">
            <span className="alert-label">STATUS:</span>
            <span className={`alert-status ${riskAssessment.level.toLowerCase()}`}>
              {riskAssessment.level === 'OK' ? '✓ OK - NO THREAT' : riskAssessment.globalStatus}
            </span>
          </div>
          {riskAssessment.tsunamiCount > 0 && (
            <div className="alert-eta">
              ACTIVE WARNINGS: <strong>{riskAssessment.tsunamiCount}</strong>
            </div>
          )}
          {riskAssessment.majorEventCount > 0 && riskAssessment.level !== 'OK' && (
            <div className="alert-regions">
              SIGNIFICANT EVENTS: {riskAssessment.majorEventCount}
            </div>
          )}
        </div>
        {sirenPlaying && (
          <button className="stop-siren-btn" onClick={stopSiren}>
            STOP ALERT
          </button>
        )}

        {/* Mute Button */}
        <button className={`mute-btn ${muted ? 'muted' : ''}`} onClick={toggleMute}>
          {muted ? '🔇 MUTED' : '🔊 SOUND ON'}
        </button>

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
                      <span className="sim-name">2004 Sumatra</span>
                      <span className="sim-desc">Deadliest tsunami in recorded history</span>
                    </div>
                  </button>
                  <button onClick={() => { startSimulation(1); setSimMenuOpen(false); }} className="sim-option">
                    <span className="sim-mag">M9.1</span>
                    <div className="sim-details">
                      <span className="sim-name">2011 Japan Tōhoku</span>
                      <span className="sim-desc">Triggered Fukushima nuclear disaster</span>
                    </div>
                  </button>
                  <button onClick={() => { startSimulation(2); setSimMenuOpen(false); }} className="sim-option">
                    <span className="sim-mag">M9.0</span>
                    <div className="sim-details">
                      <span className="sim-name">Cascadia Subduction</span>
                      <span className="sim-desc">US/Canada Pacific coast threat</span>
                    </div>
                  </button>
                  <button onClick={() => { startSimulation(3); setSimMenuOpen(false); }} className="sim-option">
                    <span className="sim-mag">M9.5</span>
                    <div className="sim-details">
                      <span className="sim-name">1960 Chile Valdivia</span>
                      <span className="sim-desc">Largest earthquake ever recorded</span>
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
              <span className="section-title">COASTAL RISK ZONES</span>
            </div>
            <div className="risk-grid">
              {GLOBAL_COASTAL_REGIONS.map(region => (
                <div
                  key={region.name}
                  className="risk-item"
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
              <span className="section-title">TSUNAMI BASINS</span>
            </div>
            <div className="region-list">
              {TSUNAMI_BASINS.map(region => (
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
                <span className="metric-value">{liveEvents.filter(e => e.magnitude >= 5).length}</span>
                <span className="metric-label">M5+ Events</span>
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
              longitude: 0,
              latitude: 20,
              zoom: 1.5,
            }}
            style={{ width: '100%', height: '100%' }}
            mapStyle="mapbox://styles/mapbox/dark-v11"
            mapboxAccessToken={mapboxToken}
          >
            <NavigationControl position="top-right" />

            {/* Tsunami Warning Impact Zones (only for tsunami events) */}
            <Source id="tsunami-impact" type="geojson" data={tsunamiImpactGeoJSON}>
              <Layer
                id="tsunami-impact-fill"
                type="fill"
                paint={{
                  'fill-color': [
                    'case',
                    ['==', ['get', 'zoneType'], 'severe'], '#dc2626',
                    ['==', ['get', 'zoneType'], 'moderate'], '#fb923c',
                    '#facc15'
                  ],
                  'fill-opacity': ['get', 'opacity']
                }}
              />
              <Layer
                id="tsunami-impact-line"
                type="line"
                paint={{
                  'line-color': [
                    'case',
                    ['==', ['get', 'zoneType'], 'severe'], 'rgba(220, 38, 38, 0.6)',
                    ['==', ['get', 'zoneType'], 'moderate'], 'rgba(251, 146, 60, 0.5)',
                    'rgba(250, 204, 21, 0.4)'
                  ],
                  'line-width': 2,
                  'line-dasharray': [4, 2]
                }}
              />
            </Source>

            {/* Geographic Wave Propagation Circles (zoom-independent) - only shown during simulations */}
            <Source id="wave-circles" type="geojson" data={waveCirclesGeoJSON}>
              <Layer
                id="wave-circles-fill"
                type="fill"
                paint={{
                  'fill-color': '#dc2626',
                  'fill-opacity': ['*', ['get', 'opacity'], 0.3]
                }}
              />
              <Layer
                id="wave-circles-line"
                type="line"
                paint={{
                  'line-color': '#dc2626',
                  'line-width': 2,
                  'line-opacity': ['get', 'opacity']
                }}
              />
            </Source>

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

            {/* Simulated Event Epicenter (wave rings are now GeoJSON layers) */}
            {simulatedEvent && (
              <Marker
                longitude={simulatedEvent.lng}
                latitude={simulatedEvent.lat}
                anchor="center"
              >
                <div className="simulated-epicenter">
                  <div className="epicenter-core">
                    <span className="epicenter-mag">M{simulatedEvent.magnitude}</span>
                  </div>
                  <div className="epicenter-pulse" />
                </div>
              </Marker>
            )}

            {/* Earthquake Markers */}
            {liveEvents.slice(0, 20).map(event => {
              const size = Math.max(20, Math.min(50, event.magnitude * 7));

              return (
                <Marker
                  key={event.event_id}
                  longitude={event.lng}
                  latitude={event.lat}
                  anchor="center"
                >
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
                      <span className="row-label">Coordinates</span>
                      <span className="row-value">{selectedEvent.lat.toFixed(2)}°, {selectedEvent.lng.toFixed(2)}°</span>
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
                  <span>No significant threats detected globally.</span>
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
                      <span>Coords: {event.lat.toFixed(1)}°, {event.lng.toFixed(1)}°</span>
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
                <span>Earthquake (size = magnitude)</span>
              </div>
              <div className="legend-item">
                <span className="legend-marker tsunami-legend" />
                <span>Tsunami Warning</span>
              </div>
              <div className="legend-item">
                <span className="legend-marker io-legend" />
                <span>Significant Event (M6+)</span>
              </div>
              <div className="legend-item">
                <span className="legend-marker impact-legend" />
                <span>Tsunami Impact Zone</span>
              </div>
              <div className="legend-item">
                <span className="legend-marker wave-legend" />
                <span>Wave Propagation (simulation)</span>
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

        .mute-btn {
          padding: 8px 14px;
          background: linear-gradient(135deg, #374151 0%, #1f2937 100%);
          border: 1px solid #4b5563;
          border-radius: 4px;
          color: #10b981;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 11px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
          margin-right: 12px;
        }

        .mute-btn:hover {
          background: linear-gradient(135deg, #4b5563 0%, #374151 100%);
        }

        .mute-btn.muted {
          color: #ef4444;
          border-color: #ef4444;
          background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
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

        .impact-legend {
          background: linear-gradient(135deg, rgba(220, 38, 38, 0.3) 0%, rgba(251, 146, 60, 0.2) 50%, rgba(250, 204, 21, 0.15) 100%);
          border: 2px dashed #fb923c;
        }

        .wave-legend {
          background: transparent;
          border: 2px solid #dc2626;
          box-shadow: 0 0 6px rgba(220, 38, 38, 0.5);
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
