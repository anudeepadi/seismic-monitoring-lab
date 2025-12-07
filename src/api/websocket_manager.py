"""
WebSocket Manager for Real-Time Training Updates

Manages WebSocket connections and broadcasts training metrics,
visualizations, and status updates to connected clients.
"""

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates.

    Features:
    - Multiple clients per training job
    - Broadcast to all clients or specific jobs
    - Connection health monitoring
    - Message queuing for offline clients
    """

    def __init__(self):
        # Job ID -> Set of connected WebSockets
        self.connections: Dict[str, Set[WebSocket]] = defaultdict(set)

        # Global connections (not tied to specific job)
        self.global_connections: Set[WebSocket] = set()

        # Message queue for reliability
        self.message_queue: Dict[str, List[Dict]] = defaultdict(list)

        # Connection metadata
        self.connection_info: Dict[WebSocket, Dict] = {}

    async def connect(
        self,
        websocket: WebSocket,
        job_id: Optional[str] = None,
    ) -> None:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: WebSocket connection
            job_id: Optional job ID to subscribe to
        """
        await websocket.accept()

        if job_id:
            self.connections[job_id].add(websocket)
        else:
            self.global_connections.add(websocket)

        self.connection_info[websocket] = {
            "connected_at": datetime.utcnow().isoformat(),
            "job_id": job_id,
            "messages_sent": 0,
        }

        logger.info(f"WebSocket connected for job {job_id or 'global'}")

        # Send any queued messages
        if job_id and job_id in self.message_queue:
            for message in self.message_queue[job_id]:
                await self._send_safe(websocket, message)
            self.message_queue[job_id].clear()

    def disconnect(
        self,
        websocket: WebSocket,
        job_id: Optional[str] = None,
    ) -> None:
        """
        Handle WebSocket disconnection.

        Args:
            websocket: WebSocket connection
            job_id: Optional job ID
        """
        if job_id and job_id in self.connections:
            self.connections[job_id].discard(websocket)
            if not self.connections[job_id]:
                del self.connections[job_id]

        self.global_connections.discard(websocket)

        if websocket in self.connection_info:
            del self.connection_info[websocket]

        logger.info(f"WebSocket disconnected for job {job_id or 'global'}")

    async def broadcast_to_job(
        self,
        job_id: str,
        message: Dict[str, Any],
    ) -> None:
        """
        Broadcast message to all clients subscribed to a job.

        Args:
            job_id: Job ID
            message: Message to send
        """
        if job_id not in self.connections:
            # Queue message if no clients connected
            self.message_queue[job_id].append(message)
            if len(self.message_queue[job_id]) > 100:
                self.message_queue[job_id] = self.message_queue[job_id][-100:]
            return

        disconnected = set()

        for websocket in self.connections[job_id]:
            success = await self._send_safe(websocket, message)
            if not success:
                disconnected.add(websocket)

        # Clean up disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket, job_id)

    async def broadcast_global(
        self,
        message: Dict[str, Any],
    ) -> None:
        """
        Broadcast message to all global connections.

        Args:
            message: Message to send
        """
        disconnected = set()

        for websocket in self.global_connections:
            success = await self._send_safe(websocket, message)
            if not success:
                disconnected.add(websocket)

        for websocket in disconnected:
            self.disconnect(websocket)

    async def send_training_update(
        self,
        job_id: str,
        epoch: int,
        metrics: Dict[str, float],
        learning_rate: float,
    ) -> None:
        """
        Send training progress update.

        Args:
            job_id: Training job ID
            epoch: Current epoch
            metrics: Training metrics
            learning_rate: Current learning rate
        """
        message = {
            "type": "training_update",
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": job_id,
            "epoch": epoch,
            "metrics": metrics,
            "learning_rate": learning_rate,
        }

        await self.broadcast_to_job(job_id, message)

    async def send_loss_components(
        self,
        job_id: str,
        data_loss: float,
        physics_loss: float,
        boundary_loss: float,
        initial_loss: float,
    ) -> None:
        """
        Send individual loss component values.

        Args:
            job_id: Training job ID
            data_loss: Data fidelity loss
            physics_loss: PDE residual loss
            boundary_loss: Boundary condition loss
            initial_loss: Initial condition loss
        """
        message = {
            "type": "loss_components",
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": job_id,
            "data_loss": data_loss,
            "physics_loss": physics_loss,
            "boundary_loss": boundary_loss,
            "initial_loss": initial_loss,
        }

        await self.broadcast_to_job(job_id, message)

    async def send_visualization_frame(
        self,
        job_id: str,
        frame_type: str,
        data: Any,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Send visualization frame (e.g., wavefield snapshot).

        Args:
            job_id: Job ID
            frame_type: Type of visualization
            data: Visualization data
            metadata: Additional metadata
        """
        message = {
            "type": "visualization_frame",
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": job_id,
            "frame_type": frame_type,
            "data": data,
            "metadata": metadata,
        }

        await self.broadcast_to_job(job_id, message)

    async def send_status_update(
        self,
        job_id: str,
        status: str,
        details: Optional[Dict] = None,
    ) -> None:
        """
        Send job status update.

        Args:
            job_id: Job ID
            status: Status string (e.g., 'running', 'completed', 'failed')
            details: Optional details
        """
        message = {
            "type": "status_update",
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": job_id,
            "status": status,
            "details": details or {},
        }

        await self.broadcast_to_job(job_id, message)

    async def send_error(
        self,
        job_id: str,
        error: str,
        details: Optional[Dict] = None,
    ) -> None:
        """
        Send error notification.

        Args:
            job_id: Job ID
            error: Error message
            details: Error details
        """
        message = {
            "type": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": job_id,
            "error": error,
            "details": details or {},
        }

        await self.broadcast_to_job(job_id, message)

    async def _send_safe(
        self,
        websocket: WebSocket,
        message: Dict[str, Any],
    ) -> bool:
        """
        Safely send message to WebSocket.

        Returns:
            True if successful, False otherwise
        """
        try:
            await websocket.send_json(message)

            if websocket in self.connection_info:
                self.connection_info[websocket]["messages_sent"] += 1

            return True

        except Exception as e:
            logger.warning(f"Failed to send WebSocket message: {e}")
            return False

    def get_connection_count(self, job_id: Optional[str] = None) -> int:
        """Get number of active connections."""
        if job_id:
            return len(self.connections.get(job_id, set()))
        return sum(len(s) for s in self.connections.values()) + len(
            self.global_connections
        )

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": self.get_connection_count(),
            "global_connections": len(self.global_connections),
            "jobs_with_connections": len(self.connections),
            "queued_messages": sum(
                len(msgs) for msgs in self.message_queue.values()
            ),
        }


class TrainingProgressEmitter:
    """
    Helper class to emit training progress updates.

    Integrates with the training loop to send real-time updates.
    """

    def __init__(
        self,
        websocket_manager: WebSocketManager,
        job_id: str,
        update_frequency: int = 1,
    ):
        self.ws_manager = websocket_manager
        self.job_id = job_id
        self.update_frequency = update_frequency
        self.step_count = 0

    async def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
        learning_rate: float,
    ) -> None:
        """Called at end of each epoch."""
        self.step_count += 1

        if self.step_count % self.update_frequency == 0:
            await self.ws_manager.send_training_update(
                self.job_id,
                epoch,
                metrics,
                learning_rate,
            )

    async def on_visualization_ready(
        self,
        frame_type: str,
        data: Any,
        metadata: Dict,
    ) -> None:
        """Called when visualization is ready."""
        await self.ws_manager.send_visualization_frame(
            self.job_id,
            frame_type,
            data,
            metadata,
        )

    async def on_training_complete(
        self,
        final_metrics: Dict[str, float],
    ) -> None:
        """Called when training completes."""
        await self.ws_manager.send_status_update(
            self.job_id,
            "completed",
            {"final_metrics": final_metrics},
        )

    async def on_error(self, error: Exception) -> None:
        """Called on training error."""
        await self.ws_manager.send_error(
            self.job_id,
            str(error),
            {"type": type(error).__name__},
        )
