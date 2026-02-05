"""
Communication Layer for Agent-to-Agent Communication.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages agents can send."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    HEARTBEAT = "heartbeat"
    BROADCAST = "broadcast"
    CONSENSUS_VOTE = "consensus_vote"
    ALERT = "alert"

@dataclass
class Message:
    """A message sent between agents."""
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: float
    priority: int = 1

class CommunicationLayer:
    """
    Manages communication between agents in a swarm.
    """
    
    def __init__(self, swarm_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the communication layer.
        
        Args:
            swarm_id: ID of the swarm
            config: Configuration dictionary
        """
        self.swarm_id = swarm_id
        self.config = config or {}
        
        # Message queues for each agent
        self.agent_queues: Dict[str, asyncio.Queue] = {}
        
        # Broadcast subscribers
        self.broadcast_subscribers: Dict[str, Callable] = {}
        
        # Message history for audit
        self.message_history: List[Message] = []
        self.max_history = self.config.get('max_history', 1000)
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'broadcasts_sent': 0,
            'average_latency': 0.0
        }
        
        logger.info(f"CommunicationLayer initialized for swarm: {swarm_id}")
    
    async def register_agent(self, agent_id: str):
        """
        Register an agent to the communication layer.
        
        Args:
            agent_id: ID of the agent to register
        """
        if agent_id not in self.agent_queues:
            self.agent_queues[agent_id] = asyncio.Queue()
            logger.info(f"Registered agent {agent_id} to communication layer")
    
    async def unregister_agent(self, agent_id: str):
        """
        Unregister an agent from the communication layer.
        
        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id in self.agent_queues:
            del self.agent_queues[agent_id]
            logger.info(f"Unregistered agent {agent_id} from communication layer")
    
    async def send_message(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: int = 1
    ) -> bool:
        """
        Send a message from one agent to another.
        
        Args:
            sender_id: ID of the sending agent
            receiver_id: ID of the receiving agent
            message_type: Type of message
            payload: Message payload
            priority: Message priority (higher = more important)
            
        Returns:
            True if message was sent successfully
        """
        if receiver_id not in self.agent_queues:
            logger.warning(f"Receiver {receiver_id} not registered")
            return False
        
        message = Message(
            message_id=f"msg_{time.time()}_{sender_id}",
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            timestamp=time.time(),
            priority=priority
        )
        
        # Add to receiver's queue
        await self.agent_queues[receiver_id].put(message)
        
        # Log message
        self._log_message(message)
        
        self.stats['messages_sent'] += 1
        logger.debug(f"Message sent: {sender_id} -> {receiver_id} ({message_type.value})")
        
        return True
    
    async def broadcast_message(
        self,
        sender_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: int = 1
    ):
        """
        Broadcast a message to all agents.
        
        Args:
            sender_id: ID of the sending agent
            message_type: Type of message
            payload: Message payload
            priority: Message priority
        """
        message = Message(
            message_id=f"broadcast_{time.time()}_{sender_id}",
            sender_id=sender_id,
            receiver_id=None,
            message_type=message_type,
            payload=payload,
            timestamp=time.time(),
            priority=priority
        )
        
        # Send to all registered agents
        for agent_id, queue in self.agent_queues.items():
            if agent_id != sender_id:  # Don't send to sender
                await queue.put(message)
        
        # Log message
        self._log_message(message)
        
        self.stats['broadcasts_sent'] += 1
        logger.info(f"Broadcast sent from {sender_id} to {len(self.agent_queues) - 1} agents")
    
    async def receive_message(
        self,
        agent_id: str,
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """
        Receive a message for an agent.
        
        Args:
            agent_id: ID of the receiving agent
            timeout: Maximum time to wait for a message
            
        Returns:
            Message if available, None if timeout
        """
        if agent_id not in self.agent_queues:
            logger.warning(f"Agent {agent_id} not registered")
            return None
        
        try:
            if timeout:
                message = await asyncio.wait_for(
                    self.agent_queues[agent_id].get(),
                    timeout=timeout
                )
            else:
                message = await self.agent_queues[agent_id].get()
            
            self.stats['messages_received'] += 1
            return message
            
        except asyncio.TimeoutError:
            return None
    
    async def get_all_messages(self, agent_id: str) -> List[Message]:
        """
        Get all pending messages for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of messages
        """
        messages = []
        
        if agent_id not in self.agent_queues:
            return messages
        
        queue = self.agent_queues[agent_id]
        
        while not queue.empty():
            try:
                message = queue.get_nowait()
                messages.append(message)
            except asyncio.QueueEmpty:
                break
        
        return messages
    
    def _log_message(self, message: Message):
        """Log a message to history."""
        self.message_history.append(message)
        
        # Trim history if too large
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return self.stats.copy()
    
    def get_message_history(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Message]:
        """
        Get message history.
        
        Args:
            agent_id: Filter by agent ID (sender or receiver)
            limit: Maximum number of messages to return
            
        Returns:
            List of messages
        """
        if agent_id:
            filtered = [
                msg for msg in self.message_history
                if msg.sender_id == agent_id or msg.receiver_id == agent_id
            ]
            return filtered[-limit:]
        else:
            return self.message_history[-limit:]
