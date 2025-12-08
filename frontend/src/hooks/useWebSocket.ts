import { useEffect, useRef, useCallback } from 'react';
import { useTrainingStore } from '../stores/trainingStore';
import config from '../config';

interface WebSocketMessage {
  type: 'progress' | 'status' | 'error' | 'completed';
  data: Record<string, unknown>;
}

interface UseWebSocketOptions {
  jobId: string | null;
  onMessage?: (message: WebSocketMessage) => void;
  onError?: (error: Event) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

export function useWebSocket({
  jobId,
  onMessage,
  onError,
  onConnect,
  onDisconnect,
}: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { setWsConnected, updateProgress, updateJob } = useTrainingStore();

  const connect = useCallback(() => {
    if (!jobId) return;

    const wsUrl = `${config.wsBase}/ws/training/${jobId}`;

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setWsConnected(true);
        onConnect?.();
      };

      ws.onclose = () => {
        setWsConnected(false);
        onDisconnect?.();

        // Attempt reconnection after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          if (jobId) {
            connect();
          }
        }, 3000);
      };

      ws.onerror = (event) => {
        onError?.(event);
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          onMessage?.(message);

          switch (message.type) {
            case 'progress':
              updateProgress(message.data as never);
              break;
            case 'status':
              updateJob(jobId, { status: message.data.status as never });
              break;
            case 'completed':
              updateJob(jobId, { status: 'completed', completed_at: new Date().toISOString() });
              break;
            case 'error':
              updateJob(jobId, { status: 'failed', error: message.data.error as string });
              break;
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('Failed to connect WebSocket:', err);
    }
  }, [jobId, onConnect, onDisconnect, onError, onMessage, setWsConnected, updateJob, updateProgress]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const sendMessage = useCallback((message: Record<string, unknown>) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  useEffect(() => {
    if (jobId) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [jobId, connect, disconnect]);

  return { sendMessage, disconnect, isConnected: wsRef.current?.readyState === WebSocket.OPEN };
}
