import { useState, useEffect, useCallback, useRef } from 'react';
import config from '../config';

const API_BASE = config.apiBaseV1;

export interface Station {
  id: string;
  network: string;
  name: string;
  country: string;
  lat: number;
  lng: number;
  status: string;
  last_update: string | null;
  amplitude: number | null;
}

export interface SeismicEvent {
  event_id: string;
  timestamp: string;
  lat: number;
  lng: number;
  magnitude: number;
  depth: number;
  place: string;
  tsunami: boolean;
  in_region: boolean;
}

export interface StreamStatus {
  is_streaming: boolean;
  last_fetch_time: string | null;
  active_connections: number;
  stations_online: number;
  recent_events: number;
}

export function useSeismicStream() {
  const [stations, setStations] = useState<Station[]>([]);
  const [events, setEvents] = useState<SeismicEvent[]>([]);
  const [indianOceanEvents, setIndianOceanEvents] = useState<SeismicEvent[]>([]);
  const [status, setStatus] = useState<StreamStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Fetch stations
  const fetchStations = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/streams/stations`);
      if (!response.ok) throw new Error('Failed to fetch stations');
      const data = await response.json();
      setStations(data);
    } catch (err) {
      console.error('Error fetching stations:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch stations');
    }
  }, []);

  // Fetch events from USGS
  const fetchEvents = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/streams/events`);
      if (!response.ok) throw new Error('Failed to fetch events');
      const data = await response.json();
      setEvents(data.events || []);
      setIndianOceanEvents(data.indian_ocean_only || []);
    } catch (err) {
      console.error('Error fetching events:', err);
    }
  }, []);

  // Fetch stream status
  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/streams/status`);
      if (!response.ok) throw new Error('Failed to fetch status');
      const data = await response.json();
      setStatus(data);
    } catch (err) {
      console.error('Error fetching status:', err);
    }
  }, []);

  // Connect to WebSocket for live updates
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const ws = new WebSocket(config.wsLive);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
      setError(null);

      // Subscribe to stations
      ws.send(JSON.stringify({
        type: 'subscribe',
        stations: ['PALK', 'COCO', 'DGAR', 'CHTO', 'TATO', 'NWAO']
      }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'update' && data.data) {
          // Update station amplitudes
          setStations(prev => prev.map(station => {
            const update = data.data.find((u: any) => u.station_id === station.id);
            if (update) {
              return {
                ...station,
                amplitude: update.amplitude,
                status: update.status,
              };
            }
            return station;
          }));
        } else if (data.type === 'event') {
          // New seismic event detected
          setEvents(prev => [data.event, ...prev].slice(0, 50));
          if (data.event.in_region) {
            setIndianOceanEvents(prev => [data.event, ...prev].slice(0, 20));
          }
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    ws.onerror = (event) => {
      console.error('WebSocket error:', event);
      setError('WebSocket connection error');
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);

      // Attempt reconnect after 5 seconds
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log('Attempting to reconnect...');
        connectWebSocket();
      }, 5000);
    };

    wsRef.current = ws;
  }, []);

  // Disconnect WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  // Initial data fetch
  useEffect(() => {
    const init = async () => {
      setLoading(true);
      await Promise.all([
        fetchStations(),
        fetchEvents(),
        fetchStatus(),
      ]);
      setLoading(false);
    };

    init();

    // Poll events every 30 seconds
    const eventInterval = setInterval(fetchEvents, 30000);

    // Poll status every 10 seconds
    const statusInterval = setInterval(fetchStatus, 10000);

    return () => {
      clearInterval(eventInterval);
      clearInterval(statusInterval);
    };
  }, [fetchStations, fetchEvents, fetchStatus]);

  // Connect to WebSocket
  useEffect(() => {
    connectWebSocket();

    return () => {
      disconnect();
    };
  }, [connectWebSocket, disconnect]);

  return {
    stations,
    events,
    indianOceanEvents,
    status,
    loading,
    error,
    connected,
    refetch: fetchStations,
    refetchEvents: fetchEvents,
  };
}

// Hook for fetching waveform data for a specific station
export function useWaveformData(stationId: string | null, minutes: number = 5) {
  const [waveform, setWaveform] = useState<{
    data: number[];
    sampleRate: number;
    amplitude: number;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchWaveform = useCallback(async () => {
    if (!stationId) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE}/streams/waveforms/${stationId}?minutes=${minutes}`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch waveform');
      }

      const data = await response.json();
      setWaveform({
        data: data.data,
        sampleRate: data.sample_rate,
        amplitude: data.amplitude,
      });
    } catch (err) {
      console.error('Error fetching waveform:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch waveform');
    } finally {
      setLoading(false);
    }
  }, [stationId, minutes]);

  useEffect(() => {
    if (stationId) {
      fetchWaveform();
    }
  }, [stationId, fetchWaveform]);

  return { waveform, loading, error, refetch: fetchWaveform };
}
