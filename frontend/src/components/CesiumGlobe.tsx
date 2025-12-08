import { useRef, useEffect, useCallback } from 'react';
import * as Cesium from 'cesium';

// No Cesium Ion token needed - using OpenStreetMap imagery

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

interface SeismicEvent {
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

interface CesiumGlobeProps {
  stations: Station[];
  events: SeismicEvent[];
  simulatedEvent: SeismicEvent | null;
  waveRadius: number;
  onStationClick?: (station: Station) => void;
  onEventClick?: (event: SeismicEvent) => void;
}

export default function CesiumGlobe({
  stations,
  events,
  simulatedEvent,
  waveRadius,
  onStationClick,
  onEventClick,
}: CesiumGlobeProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Cesium.Viewer | null>(null);
  const entitiesRef = useRef<Map<string, Cesium.Entity>>(new Map());

  // Initialize Cesium Viewer
  useEffect(() => {
    if (!containerRef.current || viewerRef.current) return;

    const viewer = new Cesium.Viewer(containerRef.current, {
      animation: false,
      baseLayerPicker: false,
      fullscreenButton: false,
      vrButton: false,
      geocoder: false,
      homeButton: false,
      infoBox: false,
      sceneModePicker: false,
      selectionIndicator: false,
      timeline: false,
      navigationHelpButton: false,
      scene3DOnly: true,
      creditContainer: document.createElement('div'),
    });

    viewerRef.current = viewer;

    // Remove default imagery and add OpenStreetMap (no token required)
    viewer.imageryLayers.removeAll();
    viewer.imageryLayers.addImageryProvider(
      new Cesium.OpenStreetMapImageryProvider({
        url: 'https://tile.openstreetmap.org/',
      })
    );

    // Set initial camera view to Indian Ocean
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(85, 5, 15000000),
      orientation: {
        heading: Cesium.Math.toRadians(0),
        pitch: Cesium.Math.toRadians(-90),
        roll: 0,
      },
      duration: 2,
    });

    // Enable lighting
    viewer.scene.globe.enableLighting = true;

    // Set atmosphere if available
    if (viewer.scene.skyAtmosphere) {
      viewer.scene.skyAtmosphere.show = true;
    }

    // Add Indian Ocean Region Boundary
    viewer.entities.add({
      polyline: {
        positions: Cesium.Cartesian3.fromDegreesArray([
          30, -40,
          30, 30,
          130, 30,
          130, -40,
          30, -40,
        ]),
        width: 2,
        material: Cesium.Color.CYAN.withAlpha(0.3),
      },
    });

    // Cleanup on unmount
    return () => {
      if (viewerRef.current && !viewerRef.current.isDestroyed()) {
        viewerRef.current.destroy();
        viewerRef.current = null;
      }
    };
  }, []);

  // Update stations
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Remove old station entities
    entitiesRef.current.forEach((entity, id) => {
      if (id.startsWith('station-')) {
        viewer.entities.remove(entity);
        entitiesRef.current.delete(id);
      }
    });

    // Add station entities
    stations.forEach((station) => {
      const color = station.status === 'offline'
        ? Cesium.Color.RED
        : station.amplitude && station.amplitude > 0.6
          ? Cesium.Color.ORANGE
          : Cesium.Color.fromCssColorString('#10b981');

      const entity = viewer.entities.add({
        id: `station-${station.id}`,
        position: Cesium.Cartesian3.fromDegrees(station.lng, station.lat, 0),
        name: station.id,
        description: `${station.name}, ${station.country}`,
        point: {
          pixelSize: 12,
          color: color,
          outlineColor: Cesium.Color.WHITE,
          outlineWidth: 2,
        },
      });

      entitiesRef.current.set(`station-${station.id}`, entity);
    });
  }, [stations]);

  // Update events
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Remove old event entities
    entitiesRef.current.forEach((entity, id) => {
      if (id.startsWith('event-')) {
        viewer.entities.remove(entity);
        entitiesRef.current.delete(id);
      }
    });

    // Add event entities
    events.slice(0, 50).forEach((event) => {
      const size = Math.max(8, Math.min(30, event.magnitude * 4));
      const color = event.tsunami
        ? Cesium.Color.RED
        : event.magnitude >= 7.0
          ? Cesium.Color.RED
          : event.magnitude >= 6.0
            ? Cesium.Color.ORANGE
            : event.magnitude >= 5.0
              ? Cesium.Color.YELLOW
              : Cesium.Color.fromCssColorString('#ef4444');

      const entity = viewer.entities.add({
        id: `event-${event.event_id}`,
        position: Cesium.Cartesian3.fromDegrees(event.lng, event.lat, 0),
        name: `M${event.magnitude.toFixed(1)}`,
        description: event.place,
        point: {
          pixelSize: size,
          color: color,
          outlineColor: event.in_region ? Cesium.Color.CYAN : Cesium.Color.WHITE,
          outlineWidth: event.in_region ? 3 : 1,
        },
      });

      entitiesRef.current.set(`event-${event.event_id}`, entity);

      // Add impact radius for M5.0+ events
      if (event.magnitude >= 5.0) {
        const impactRadius = Math.pow(10, (event.magnitude - 3) / 1.5) * 15 * 1000;
        const ellipseEntity = viewer.entities.add({
          id: `event-ellipse-${event.event_id}`,
          position: Cesium.Cartesian3.fromDegrees(event.lng, event.lat),
          ellipse: {
            semiMajorAxis: impactRadius * 0.3,
            semiMinorAxis: impactRadius * 0.3,
            material: color.withAlpha(0.2),
            outline: true,
            outlineColor: color.withAlpha(0.5),
            outlineWidth: 1,
            height: 0,
          },
        });
        entitiesRef.current.set(`event-ellipse-${event.event_id}`, ellipseEntity);
      }
    });
  }, [events]);

  // Update simulated event and wave rings
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Remove old simulation entities
    entitiesRef.current.forEach((entity, id) => {
      if (id.startsWith('sim-')) {
        viewer.entities.remove(entity);
        entitiesRef.current.delete(id);
      }
    });

    if (!simulatedEvent) return;

    // Add simulated event epicenter
    const epicenterEntity = viewer.entities.add({
      id: 'sim-epicenter',
      position: Cesium.Cartesian3.fromDegrees(simulatedEvent.lng, simulatedEvent.lat, 0),
      name: `SIMULATION: M${simulatedEvent.magnitude}`,
      description: simulatedEvent.place,
      point: {
        pixelSize: 40,
        color: Cesium.Color.RED,
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 4,
      },
    });
    entitiesRef.current.set('sim-epicenter', epicenterEntity);

    // Add wave propagation rings
    if (waveRadius > 0) {
      const numRings = 5;
      for (let i = 0; i < numRings; i++) {
        const ringRadius = (waveRadius * 1000) * (1 - i * 0.15);
        if (ringRadius <= 0) continue;

        const opacity = 0.6 - i * 0.1;
        const ringEntity = viewer.entities.add({
          id: `sim-ring-${i}`,
          position: Cesium.Cartesian3.fromDegrees(simulatedEvent.lng, simulatedEvent.lat),
          ellipse: {
            semiMajorAxis: ringRadius,
            semiMinorAxis: ringRadius,
            material: Cesium.Color.RED.withAlpha(opacity * 0.3),
            outline: true,
            outlineColor: Cesium.Color.RED.withAlpha(opacity),
            outlineWidth: 2,
            height: 0,
          },
        });
        entitiesRef.current.set(`sim-ring-${i}`, ringEntity);
      }
    }
  }, [simulatedEvent, waveRadius]);

  // Handle click events
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);

    handler.setInputAction((click: { position: Cesium.Cartesian2 }) => {
      const pickedObject = viewer.scene.pick(click.position);
      if (Cesium.defined(pickedObject) && pickedObject.id) {
        const entityId = pickedObject.id.id as string;

        if (entityId.startsWith('station-')) {
          const stationId = entityId.replace('station-', '');
          const station = stations.find(s => s.id === stationId);
          if (station && onStationClick) {
            onStationClick(station);
          }
        } else if (entityId.startsWith('event-') && !entityId.includes('ellipse')) {
          const eventId = entityId.replace('event-', '');
          const event = events.find(e => e.event_id === eventId);
          if (event && onEventClick) {
            onEventClick(event);
          }
        }
      }
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

    return () => {
      handler.destroy();
    };
  }, [stations, events, onStationClick, onEventClick]);

  return (
    <>
      <div ref={containerRef} className="cesium-globe-container" />
      <style>{`
        .cesium-globe-container {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          overflow: hidden;
        }

        .cesium-viewer-bottom,
        .cesium-widget-credits {
          display: none !important;
        }
      `}</style>
    </>
  );
}
