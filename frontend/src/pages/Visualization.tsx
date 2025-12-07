import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import {
  Waves,
  Mountain,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Maximize2,
  Download,
  Sliders,
  Eye,
} from 'lucide-react';
import * as THREE from 'three';

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

// Generate synthetic velocity model data (Marmousi-like)
function generateVelocityModel(nx: number, nz: number): number[][] {
  const velocity = Array(nz)
    .fill(0)
    .map(() => Array(nx).fill(0));

  for (let z = 0; z < nz; z++) {
    for (let x = 0; x < nx; x++) {
      // Base velocity increases with depth
      let v = 1500 + (z / nz) * 2500;

      // Add some layer structure
      const layerPeriod = nz / 8;
      v += Math.sin((z / layerPeriod) * Math.PI * 2) * 200;

      // Add lateral variation
      v += Math.sin((x / nx) * Math.PI * 4) * 150 * (z / nz);

      // Add some noise
      v += (Math.random() - 0.5) * 50;

      velocity[z][x] = v;
    }
  }

  return velocity;
}

// Generate synthetic wavefield data
function generateWavefield(nx: number, nz: number, time: number): number[][] {
  const wavefield = Array(nz)
    .fill(0)
    .map(() => Array(nx).fill(0));

  const sourceX = nx / 2;
  const sourceZ = 10;
  const velocity = 2500;
  const frequency = 15;
  const wavelength = velocity / frequency;

  for (let z = 0; z < nz; z++) {
    for (let x = 0; x < nx; x++) {
      const dist = Math.sqrt((x - sourceX) ** 2 + (z - sourceZ) ** 2);
      const travelTime = dist / velocity;
      const phase = 2 * Math.PI * frequency * (time - travelTime);

      if (time > travelTime) {
        // Ricker wavelet envelope
        const sigma = 1 / (Math.PI * frequency);
        const env = Math.exp(-((time - travelTime) ** 2) / (2 * sigma ** 2));
        const amplitude = env * Math.cos(phase) / (1 + dist * 0.01);

        wavefield[z][x] = amplitude;
      }
    }
  }

  return wavefield;
}

// 3D Velocity Model Visualization Component
function VelocityModelMesh({ data, colorScale }: { data: number[][]; colorScale: 'seismic' | 'viridis' | 'thermal' }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const nz = data.length;
  const nx = data[0].length;

  const geometry = useMemo(() => {
    const geo = new THREE.PlaneGeometry(10, 5, nx - 1, nz - 1);
    const positions = geo.attributes.position;
    const colors = new Float32Array(positions.count * 3);

    // Find min/max for normalization
    let minVal = Infinity, maxVal = -Infinity;
    data.forEach(row => row.forEach(v => {
      minVal = Math.min(minVal, v);
      maxVal = Math.max(maxVal, v);
    }));

    for (let i = 0; i < positions.count; i++) {
      const x = i % nx;
      const z = Math.floor(i / nx);
      const val = data[z]?.[x] || 0;
      const normalized = (val - minVal) / (maxVal - minVal);

      // Displacement for 3D effect
      positions.setZ(i, normalized * 0.5);

      // Color mapping
      let r, g, b;
      if (colorScale === 'seismic') {
        if (normalized < 0.5) {
          r = normalized * 2;
          g = normalized * 2;
          b = 1;
        } else {
          r = 1;
          g = 2 - normalized * 2;
          b = 2 - normalized * 2;
        }
      } else if (colorScale === 'viridis') {
        r = 0.267004 + normalized * 0.581 - normalized ** 2 * 0.134;
        g = 0.004874 + normalized * 0.873 - normalized ** 2 * 0.297;
        b = 0.329415 + normalized * 0.288 - normalized ** 2 * 0.788;
      } else {
        // Thermal
        r = Math.min(1, normalized * 2);
        g = normalized > 0.5 ? (normalized - 0.5) * 2 : 0;
        b = 0;
      }

      colors[i * 3] = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;
    }

    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();

    return geo;
  }, [data, colorScale, nx, nz]);

  return (
    <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 2, 0, 0]} position={[0, -1, 0]}>
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} />
    </mesh>
  );
}

// Animated wavefield visualization
function WavefieldMesh({ isPlaying, time }: { isPlaying: boolean; time: number }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [wavefield, setWavefield] = useState<number[][]>([]);
  const nx = 100;
  const nz = 50;

  useFrame((state, delta) => {
    if (isPlaying && meshRef.current) {
      const newWavefield = generateWavefield(nx, nz, time);
      setWavefield(newWavefield);
    }
  });

  useEffect(() => {
    setWavefield(generateWavefield(nx, nz, time));
  }, [time]);

  const geometry = useMemo(() => {
    if (wavefield.length === 0) return new THREE.PlaneGeometry(10, 5, nx - 1, nz - 1);

    const geo = new THREE.PlaneGeometry(10, 5, nx - 1, nz - 1);
    const positions = geo.attributes.position;
    const colors = new Float32Array(positions.count * 3);

    for (let i = 0; i < positions.count; i++) {
      const x = i % nx;
      const z = Math.floor(i / nx);
      const val = wavefield[z]?.[x] || 0;

      // Height based on amplitude
      positions.setZ(i, val * 2);

      // Seismic colormap (red-white-blue)
      const normalized = (val + 1) / 2;
      if (val < 0) {
        colors[i * 3] = 0.2 + Math.abs(val) * 0.2;
        colors[i * 3 + 1] = 0.4 + Math.abs(val) * 0.2;
        colors[i * 3 + 2] = 0.8 + Math.abs(val) * 0.2;
      } else {
        colors[i * 3] = 0.8 + val * 0.2;
        colors[i * 3 + 1] = 0.3 - val * 0.2;
        colors[i * 3 + 2] = 0.2;
      }
    }

    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();

    return geo;
  }, [wavefield]);

  return (
    <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} wireframe={false} />
    </mesh>
  );
}

// 2D Canvas visualization for velocity/wavefield
function Canvas2D({ data, colorScale, title }: { data: number[][]; colorScale: string; title: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const nz = data.length;
    const nx = data[0].length;

    // Find min/max
    let minVal = Infinity, maxVal = -Infinity;
    data.forEach(row => row.forEach(v => {
      minVal = Math.min(minVal, v);
      maxVal = Math.max(maxVal, v);
    }));

    const imageData = ctx.createImageData(nx, nz);

    for (let z = 0; z < nz; z++) {
      for (let x = 0; x < nx; x++) {
        const val = data[z][x];
        const normalized = (val - minVal) / (maxVal - minVal);
        const idx = (z * nx + x) * 4;

        if (colorScale === 'seismic') {
          if (normalized < 0.5) {
            imageData.data[idx] = Math.floor(normalized * 2 * 255);
            imageData.data[idx + 1] = Math.floor(normalized * 2 * 255);
            imageData.data[idx + 2] = 255;
          } else {
            imageData.data[idx] = 255;
            imageData.data[idx + 1] = Math.floor((2 - normalized * 2) * 255);
            imageData.data[idx + 2] = Math.floor((2 - normalized * 2) * 255);
          }
        } else {
          // Grayscale
          const gray = Math.floor(normalized * 255);
          imageData.data[idx] = gray;
          imageData.data[idx + 1] = gray;
          imageData.data[idx + 2] = gray;
        }
        imageData.data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [data, colorScale]);

  return (
    <canvas
      ref={canvasRef}
      width={data[0]?.length || 200}
      height={data.length || 100}
      className="w-full h-full object-contain"
    />
  );
}

export default function Visualization() {
  const [viewMode, setViewMode] = useState<'velocity' | 'wavefield'>('velocity');
  const [view3D, setView3D] = useState(true);
  const [colorScale, setColorScale] = useState<'seismic' | 'viridis' | 'thermal'>('viridis');
  const [isPlaying, setIsPlaying] = useState(false);
  const [time, setTime] = useState(0.1);

  // Generate data
  const velocityModel = useMemo(() => generateVelocityModel(200, 100), []);
  const wavefield = useMemo(() => generateWavefield(200, 100, time), [time]);

  // Animation loop for wavefield
  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setTime((t) => {
          const newT = t + 0.01;
          return newT > 0.5 ? 0.1 : newT;
        });
      }, 50);
      return () => clearInterval(interval);
    }
  }, [isPlaying]);

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="space-y-6"
    >
      {/* Header */}
      <motion.div variants={item} className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Visualization</h1>
          <p className="text-white/60 mt-1">
            Explore velocity models and wavefields in 2D/3D
          </p>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex bg-white/10 rounded-xl p-1">
            <button
              onClick={() => setViewMode('velocity')}
              className={`px-4 py-2 rounded-lg transition-all ${
                viewMode === 'velocity'
                  ? 'bg-primary-500 text-white'
                  : 'text-white/60 hover:text-white'
              }`}
            >
              <Mountain className="w-4 h-4 inline mr-2" />
              Velocity
            </button>
            <button
              onClick={() => setViewMode('wavefield')}
              className={`px-4 py-2 rounded-lg transition-all ${
                viewMode === 'wavefield'
                  ? 'bg-primary-500 text-white'
                  : 'text-white/60 hover:text-white'
              }`}
            >
              <Waves className="w-4 h-4 inline mr-2" />
              Wavefield
            </button>
          </div>

          <button
            onClick={() => setView3D(!view3D)}
            className={`p-2 rounded-xl transition-all ${
              view3D ? 'bg-primary-500' : 'bg-white/10'
            }`}
          >
            <Maximize2 className="w-5 h-5" />
          </button>
        </div>
      </motion.div>

      {/* Main Visualization */}
      <motion.div variants={item} className="glass-panel overflow-hidden" style={{ height: '500px' }}>
        {view3D ? (
          <Canvas>
            <PerspectiveCamera makeDefault position={[8, 6, 8]} />
            <OrbitControls enableDamping dampingFactor={0.05} />
            <ambientLight intensity={0.5} />
            <directionalLight position={[10, 10, 5]} intensity={1} />
            <pointLight position={[-10, -10, -10]} intensity={0.5} />

            {viewMode === 'velocity' ? (
              <VelocityModelMesh data={velocityModel} colorScale={colorScale} />
            ) : (
              <WavefieldMesh isPlaying={isPlaying} time={time} />
            )}

            <gridHelper args={[12, 12, 'rgba(255,255,255,0.1)', 'rgba(255,255,255,0.05)']} />
          </Canvas>
        ) : (
          <div className="w-full h-full p-4 flex items-center justify-center bg-black/20">
            <Canvas2D
              data={viewMode === 'velocity' ? velocityModel : wavefield}
              colorScale={colorScale}
              title={viewMode === 'velocity' ? 'Velocity Model' : 'Wavefield'}
            />
          </div>
        )}
      </motion.div>

      {/* Controls */}
      <motion.div variants={item} className="glass-panel p-6">
        <div className="flex items-center justify-between">
          {/* Playback Controls (for wavefield) */}
          {viewMode === 'wavefield' && (
            <div className="flex items-center gap-4">
              <button
                onClick={() => setTime(0.1)}
                className="p-2 bg-white/10 rounded-lg hover:bg-white/20 transition-all"
              >
                <SkipBack className="w-5 h-5" />
              </button>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="p-3 bg-primary-500 rounded-xl hover:bg-primary-400 transition-all"
              >
                {isPlaying ? (
                  <Pause className="w-6 h-6" />
                ) : (
                  <Play className="w-6 h-6" />
                )}
              </button>
              <button
                onClick={() => setTime((t) => Math.min(t + 0.05, 0.5))}
                className="p-2 bg-white/10 rounded-lg hover:bg-white/20 transition-all"
              >
                <SkipForward className="w-5 h-5" />
              </button>

              <div className="ml-4 flex items-center gap-2">
                <span className="text-sm text-white/60">Time:</span>
                <input
                  type="range"
                  min="0.1"
                  max="0.5"
                  step="0.01"
                  value={time}
                  onChange={(e) => setTime(parseFloat(e.target.value))}
                  className="w-32"
                />
                <span className="text-sm font-mono">{time.toFixed(3)}s</span>
              </div>
            </div>
          )}

          {viewMode === 'velocity' && <div />}

          {/* View Settings */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-white/60" />
              <span className="text-sm text-white/60">Colormap:</span>
              <select
                value={colorScale}
                onChange={(e) => setColorScale(e.target.value as typeof colorScale)}
                className="bg-white/10 border border-white/20 rounded-lg px-3 py-1 text-sm"
              >
                <option value="viridis">Viridis</option>
                <option value="seismic">Seismic</option>
                <option value="thermal">Thermal</option>
              </select>
            </div>

            <button className="btn-secondary flex items-center gap-2">
              <Download className="w-4 h-4" />
              Export
            </button>
          </div>
        </div>
      </motion.div>

      {/* Info Panel */}
      <motion.div variants={item} className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-panel p-6">
          <h3 className="font-semibold mb-2 flex items-center gap-2">
            <Eye className="w-4 h-4 text-primary-400" />
            Current View
          </h3>
          <p className="text-sm text-white/60">
            {viewMode === 'velocity'
              ? 'Synthetic velocity model with lateral variations and layer structure'
              : `Wavefield propagation at t = ${time.toFixed(3)}s from a point source`}
          </p>
        </div>

        <div className="glass-panel p-6">
          <h3 className="font-semibold mb-2">Model Dimensions</h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <span className="text-white/60">Width (X):</span>
            <span>2000 m</span>
            <span className="text-white/60">Depth (Z):</span>
            <span>1000 m</span>
            <span className="text-white/60">Grid:</span>
            <span>200 × 100</span>
          </div>
        </div>

        <div className="glass-panel p-6">
          <h3 className="font-semibold mb-2">Statistics</h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <span className="text-white/60">Min Velocity:</span>
            <span>1500 m/s</span>
            <span className="text-white/60">Max Velocity:</span>
            <span>4000 m/s</span>
            <span className="text-white/60">Source Freq:</span>
            <span>15 Hz</span>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
