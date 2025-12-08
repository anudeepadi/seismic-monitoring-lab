import { useState, useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import config from '../config';
import {
  Globe,
  Radio,
  Download,
  Play,
  Pause,
  RotateCcw,
  MapPin,
  Clock,
  Activity,
  Layers,
  AlertTriangle,
  CheckCircle,
  Loader2,
  Image,
} from 'lucide-react';

interface SeismicEvent {
  event_key: string;
  event_id: string;
  name: string;
  magnitude: number;
  latitude: number;
  longitude: number;
  depth_km: number;
  origin_time: string;
  event_type: string;
  description: string;
}

interface Station {
  network: string;
  station: string;
  latitude: number;
  longitude: number;
  samples: number;
}

interface VelocityModelData {
  status: string;
  region: string;
  shape: number[];
  velocity: number[][];
  stats: {
    min: number;
    max: number;
    mean: number;
  };
}

interface TrainingImage {
  filename: string;
  path: string;
  title: string;
  description: string;
  size_bytes: number;
}

const API_BASE = config.apiBaseV1;

export default function SeismicData() {
  const [events, setEvents] = useState<SeismicEvent[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<string>('sumatra_2004');
  const [eventInfo, setEventInfo] = useState<any>(null);
  const [stations, setStations] = useState<Station[]>([]);
  const [velocityModel, setVelocityModel] = useState<VelocityModelData | null>(null);
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [downloadStatus, setDownloadStatus] = useState<string | null>(null);
  const [trainingImages, setTrainingImages] = useState<TrainingImage[]>([]);
  const [selectedImage, setSelectedImage] = useState<TrainingImage | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const waveformCanvasRef = useRef<HTMLCanvasElement>(null);

  // Fetch training images
  const fetchTrainingImages = async () => {
    try {
      const res = await fetch(`${API_BASE}/seismic/training-results/images`);
      const data = await res.json();
      if (data.images && data.images.length > 0) {
        setTrainingImages(data.images);
        // Auto-select the summary image if available
        const summary = data.images.find((img: TrainingImage) => img.filename.includes('summary'));
        if (summary) {
          setSelectedImage(summary);
        } else {
          setSelectedImage(data.images[0]);
        }
      }
    } catch (err) {
      console.error('Failed to fetch training images:', err);
    }
  };

  // Fetch available events on mount
  useEffect(() => {
    fetch(`${API_BASE}/seismic/events`)
      .then(res => res.json())
      .then(data => {
        setEvents(data.events);
      })
      .catch(err => console.error('Failed to fetch events:', err));

    // Auto-load training images on mount (fast - local files)
    fetchTrainingImages();
  }, []);

  // Fetch event details when selected
  useEffect(() => {
    if (selectedEvent === 'sumatra_2004') {
      setLoading(prev => ({ ...prev, info: true }));
      fetch(`${API_BASE}/seismic/sumatra-2004/info`)
        .then(res => res.json())
        .then(data => {
          setEventInfo(data);
          setLoading(prev => ({ ...prev, info: false }));
        })
        .catch(err => {
          console.error('Failed to fetch event info:', err);
          setLoading(prev => ({ ...prev, info: false }));
        });
    }
  }, [selectedEvent]);

  // Fetch velocity model
  const fetchVelocityModel = async () => {
    setLoading(prev => ({ ...prev, velocity: true }));
    try {
      const res = await fetch(`${API_BASE}/seismic/velocity-model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ region: 'sumatra', nx: 100, nz: 50 }),
      });
      const data = await res.json();
      setVelocityModel(data);
      renderVelocityModel(data);
    } catch (err) {
      console.error('Failed to fetch velocity model:', err);
    }
    setLoading(prev => ({ ...prev, velocity: false }));
  };

  // Load cached waveforms (fast - from local cache)
  const loadCachedWaveforms = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/seismic/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          event_name: 'sumatra_2004',
          duration_minutes: 30,
        }),
      });
      const data = await res.json();
      if (data.status === 'cached' && data.stations) {
        setStations(data.stations);
        setDownloadStatus(`Loaded ${data.waveforms_downloaded} cached waveforms from ${data.stations.length} stations`);
        renderWaveformPreview(data.stations);
      }
    } catch (err) {
      console.error('Failed to load cached waveforms:', err);
    }
  }, []);

  // Auto-load velocity model and cached waveforms on mount
  useEffect(() => {
    fetchVelocityModel();
    loadCachedWaveforms();
  }, []);

  // Download seismic data
  const downloadData = async () => {
    setLoading(prev => ({ ...prev, download: true }));
    setDownloadStatus('Downloading waveforms from IRIS...');
    try {
      const res = await fetch(`${API_BASE}/seismic/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          event_name: selectedEvent,
          duration_minutes: 30,
        }),
      });
      const data = await res.json();
      if (data.stations) {
        setStations(data.stations);
        const statusMsg = data.status === 'cached'
          ? `Loaded ${data.waveforms_downloaded} cached waveforms from ${data.stations.length} stations`
          : `Downloaded ${data.waveforms_downloaded} waveforms from ${data.stations.length} stations`;
        setDownloadStatus(statusMsg);
        renderWaveformPreview(data.stations);
      }
    } catch (err) {
      console.error('Failed to download data:', err);
      setDownloadStatus('Download failed');
    }
    setLoading(prev => ({ ...prev, download: false }));
  };

  // Render velocity model on canvas
  const renderVelocityModel = (data: VelocityModelData) => {
    const canvas = canvasRef.current;
    if (!canvas || !data.velocity) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const nz = data.shape[0];
    const nx = data.shape[1];

    const cellWidth = width / nx;
    const cellHeight = height / nz;

    const vMin = data.stats.min;
    const vMax = data.stats.max;

    // Draw velocity model
    for (let j = 0; j < nz; j++) {
      for (let i = 0; i < nx; i++) {
        const v = data.velocity[j][i];
        const normalized = (v - vMin) / (vMax - vMin);

        // Color map: blue (slow) -> white -> red (fast)
        let r, g, b;
        if (normalized < 0.5) {
          const t = normalized * 2;
          r = Math.floor(255 * t);
          g = Math.floor(255 * t);
          b = 255;
        } else {
          const t = (normalized - 0.5) * 2;
          r = 255;
          g = Math.floor(255 * (1 - t));
          b = Math.floor(255 * (1 - t));
        }

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(i * cellWidth, j * cellHeight, cellWidth + 1, cellHeight + 1);
      }
    }

    // Draw labels
    ctx.fillStyle = 'white';
    ctx.font = '12px system-ui';
    ctx.fillText('Continental Crust', 10, 20);
    ctx.fillText('Upper Mantle', 10, height - 30);

    // Draw Moho line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(0, height * 0.15);
    ctx.lineTo(width, height * 0.15);
    ctx.stroke();
    ctx.setLineDash([]);
  };

  // Render waveform preview
  const renderWaveformPreview = (stationData: Station[]) => {
    const canvas = waveformCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, width, height);

    // Draw simulated waveforms
    const numWaveforms = Math.min(8, stationData.length);
    const waveHeight = height / numWaveforms;

    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth = 1;

    for (let i = 0; i < numWaveforms; i++) {
      const y = i * waveHeight + waveHeight / 2;
      const station = stationData[i];

      // Draw station label
      ctx.fillStyle = '#9ca3af';
      ctx.font = '10px system-ui';
      ctx.fillText(`${station.network}.${station.station}`, 5, y - waveHeight / 3);

      // Draw waveform (simulated)
      ctx.beginPath();
      ctx.moveTo(60, y);

      for (let x = 60; x < width - 10; x++) {
        const t = (x - 60) / (width - 70);
        // Simulate P-wave arrival and surface waves
        let amplitude = 0;
        if (t > 0.1 && t < 0.15) {
          amplitude = Math.sin((t - 0.1) * 100) * 0.3 * Math.exp(-(t - 0.1) * 20);
        }
        if (t > 0.2) {
          amplitude += Math.sin(t * 50) * Math.exp(-(t - 0.2) * 3) * 0.8;
        }
        if (t > 0.4) {
          amplitude += Math.sin(t * 30) * Math.exp(-(t - 0.4) * 2) * 0.5;
        }

        ctx.lineTo(x, y + amplitude * waveHeight * 0.4);
      }
      ctx.stroke();

      // Separator line
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.beginPath();
      ctx.moveTo(0, (i + 1) * waveHeight);
      ctx.lineTo(width, (i + 1) * waveHeight);
      ctx.stroke();
      ctx.strokeStyle = '#60a5fa';
    }

    // Time axis
    ctx.fillStyle = '#6b7280';
    ctx.font = '10px system-ui';
    ctx.fillText('0s', 60, height - 5);
    ctx.fillText('30min', width - 40, height - 5);
  };

  const selectedEventData = events.find(e => e.event_key === selectedEvent);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-red-400 to-orange-400 bg-clip-text text-transparent">
            Real Seismic Data
          </h1>
          <p className="text-white/60 mt-1">
            Download and visualize earthquake data from IRIS
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={downloadData}
            disabled={loading.download}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-red-500 to-orange-500 rounded-xl font-medium hover:opacity-90 transition-opacity disabled:opacity-50"
          >
            {loading.download ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Download className="w-4 h-4" />
            )}
            Download Waveforms
          </button>
        </div>
      </div>

      {/* Event Selection */}
      <div className="grid grid-cols-4 gap-4">
        {events.map(event => (
          <motion.button
            key={event.event_key}
            onClick={() => setSelectedEvent(event.event_key)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className={`p-4 rounded-xl border transition-all ${
              selectedEvent === event.event_key
                ? 'bg-red-500/20 border-red-500/50 text-red-400'
                : 'bg-white/5 border-white/10 hover:bg-white/10'
            }`}
          >
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-lg ${
                selectedEvent === event.event_key ? 'bg-red-500/20' : 'bg-white/10'
              }`}>
                <AlertTriangle className="w-5 h-5" />
              </div>
              <div className="text-left">
                <div className="font-semibold">M{event.magnitude}</div>
                <div className="text-sm text-white/60">{event.event_key}</div>
              </div>
            </div>
            <div className="mt-2 text-xs text-white/40 truncate">{event.name}</div>
          </motion.button>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-2 gap-6">
        {/* Event Info Card */}
        <div className="bg-surface-secondary/50 backdrop-blur-sm rounded-2xl border border-white/10 p-6">
          <div className="flex items-center gap-2 mb-4">
            <Globe className="w-5 h-5 text-red-400" />
            <h2 className="text-lg font-semibold">Event Information</h2>
          </div>

          {selectedEventData && (
            <div className="space-y-4">
              <div>
                <h3 className="text-xl font-bold">{selectedEventData.name}</h3>
                <p className="text-white/60 text-sm mt-1">{selectedEventData.description}</p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-white/5 rounded-xl">
                  <div className="text-2xl font-bold text-red-400">
                    M{selectedEventData.magnitude}
                  </div>
                  <div className="text-xs text-white/50">Magnitude</div>
                </div>
                <div className="p-3 bg-white/5 rounded-xl">
                  <div className="text-2xl font-bold text-orange-400">
                    {selectedEventData.depth_km} km
                  </div>
                  <div className="text-xs text-white/50">Depth</div>
                </div>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <MapPin className="w-4 h-4 text-white/40" />
                  <span className="text-white/60">
                    {selectedEventData.latitude.toFixed(3)}°N, {selectedEventData.longitude.toFixed(3)}°E
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-white/40" />
                  <span className="text-white/60">
                    {new Date(selectedEventData.origin_time).toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-white/40" />
                  <span className="text-white/60 capitalize">
                    {selectedEventData.event_type} earthquake
                  </span>
                </div>
              </div>

              {eventInfo && selectedEvent === 'sumatra_2004' && (
                <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-xl">
                  <h4 className="font-semibold text-red-400 mb-2">Tsunami Impact</h4>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div>
                      <div className="text-lg font-bold">{eventInfo.event?.tsunami?.max_wave_height_m}m</div>
                      <div className="text-xs text-white/50">Max Wave Height</div>
                    </div>
                    <div>
                      <div className="text-lg font-bold">{eventInfo.event?.tsunami?.affected_countries}</div>
                      <div className="text-xs text-white/50">Countries</div>
                    </div>
                    <div>
                      <div className="text-lg font-bold">{eventInfo.event?.rupture?.length_km}km</div>
                      <div className="text-xs text-white/50">Rupture Length</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Velocity Model Card */}
        <div className="bg-surface-secondary/50 backdrop-blur-sm rounded-2xl border border-white/10 p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Layers className="w-5 h-5 text-blue-400" />
              <h2 className="text-lg font-semibold">Velocity Model</h2>
            </div>
            <button
              onClick={fetchVelocityModel}
              disabled={loading.velocity}
              className="flex items-center gap-2 px-3 py-1.5 bg-blue-500/20 text-blue-400 rounded-lg text-sm hover:bg-blue-500/30 transition-colors disabled:opacity-50"
            >
              {loading.velocity ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <RotateCcw className="w-4 h-4" />
              )}
              Generate
            </button>
          </div>

          <div className="aspect-[2/1] bg-black/30 rounded-xl overflow-hidden relative">
            <canvas
              ref={canvasRef}
              width={400}
              height={200}
              className="w-full h-full"
            />
            {!velocityModel && (
              <div className="absolute inset-0 flex items-center justify-center text-white/40">
                Click "Generate" to load velocity model
              </div>
            )}
          </div>

          {velocityModel && (
            <div className="mt-4 flex justify-between text-sm">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-blue-500 rounded" />
                <span className="text-white/60">{velocityModel.stats.min.toFixed(1)} km/s (slow)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-red-500 rounded" />
                <span className="text-white/60">{velocityModel.stats.max.toFixed(1)} km/s (fast)</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Waveform Display */}
      <div className="bg-surface-secondary/50 backdrop-blur-sm rounded-2xl border border-white/10 p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Radio className="w-5 h-5 text-green-400" />
            <h2 className="text-lg font-semibold">Seismic Waveforms</h2>
          </div>
          {downloadStatus && (
            <div className="flex items-center gap-2 text-sm">
              {loading.download ? (
                <Loader2 className="w-4 h-4 animate-spin text-yellow-400" />
              ) : (
                <CheckCircle className="w-4 h-4 text-green-400" />
              )}
              <span className="text-white/60">{downloadStatus}</span>
            </div>
          )}
        </div>

        <div className="aspect-[3/1] bg-black/30 rounded-xl overflow-hidden relative">
          <canvas
            ref={waveformCanvasRef}
            width={900}
            height={300}
            className="w-full h-full"
          />
          {stations.length === 0 && !loading.download && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-white/40">
              <Radio className="w-8 h-8 mb-2" />
              <span>Pre-generated waveforms shown in Training Results below</span>
              <span className="text-xs mt-1">Click "Download Waveforms" for live IRIS data (slow)</span>
            </div>
          )}
          {loading.download && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-white/60">
              <Loader2 className="w-8 h-8 mb-2 animate-spin text-yellow-400" />
              <span>Downloading from IRIS servers...</span>
              <span className="text-xs mt-1 text-white/40">This may take 1-2 minutes</span>
            </div>
          )}
        </div>

        {/* Station List */}
        {stations.length > 0 && (
          <div className="mt-4 grid grid-cols-4 gap-2">
            {stations.slice(0, 8).map((station, i) => (
              <div
                key={i}
                className="p-2 bg-white/5 rounded-lg text-sm flex items-center gap-2"
              >
                <div className="w-2 h-2 rounded-full bg-green-400" />
                <span className="font-mono">{station.network}.{station.station}</span>
                <span className="text-white/40 text-xs ml-auto">
                  {(station.samples / 20 / 60).toFixed(0)}min
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Training Results Gallery */}
      {trainingImages.length > 0 && (
        <div className="bg-surface-secondary/50 backdrop-blur-sm rounded-2xl border border-white/10 p-6">
          <div className="flex items-center gap-2 mb-4">
            <Image className="w-5 h-5 text-purple-400" />
            <h2 className="text-lg font-semibold">PINN Training Results</h2>
            <span className="ml-auto text-sm text-white/50">2004 Sumatra Earthquake Analysis</span>
          </div>

          {/* Main Image Display */}
          {selectedImage && (
            <div className="mb-4">
              <div className="bg-black/40 rounded-xl p-2 overflow-hidden">
                <img
                  src={`${API_BASE}${selectedImage.path}`}
                  alt={selectedImage.title}
                  className="w-full h-auto rounded-lg"
                />
              </div>
              <div className="mt-2 text-center">
                <h3 className="font-semibold text-white/90">{selectedImage.title}</h3>
                <p className="text-sm text-white/50">{selectedImage.description}</p>
              </div>
            </div>
          )}

          {/* Image Thumbnails */}
          <div className="grid grid-cols-6 gap-3">
            {trainingImages.map((img) => (
              <motion.button
                key={img.filename}
                onClick={() => setSelectedImage(img)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className={`relative aspect-video rounded-lg overflow-hidden border-2 transition-all ${
                  selectedImage?.filename === img.filename
                    ? 'border-purple-500 ring-2 ring-purple-500/30'
                    : 'border-white/10 hover:border-white/30'
                }`}
              >
                <img
                  src={`${API_BASE}${img.path}`}
                  alt={img.title}
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                <div className="absolute bottom-1 left-1 right-1 text-xs text-white/80 truncate">
                  {img.title}
                </div>
              </motion.button>
            ))}
          </div>
        </div>
      )}

      {/* Data Sources */}
      {eventInfo && (
        <div className="grid grid-cols-3 gap-4">
          {Object.entries(eventInfo.data_sources || {}).map(([key, source]: [string, any]) => (
            <a
              key={key}
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="p-4 bg-white/5 rounded-xl border border-white/10 hover:bg-white/10 transition-colors"
            >
              <div className="font-semibold text-white/80 uppercase text-sm">{key}</div>
              <div className="text-white/50 text-sm mt-1">{source.description}</div>
            </a>
          ))}
        </div>
      )}
    </motion.div>
  );
}
