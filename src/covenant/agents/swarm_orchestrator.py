"""
Swarm orchestrator for multi-agent systems with Byzantine consensus.
"""

import asyncio
import logging
import random
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

from covenant.agents.consensus_protocol import ConsensusProtocol
from covenant.agents.agent_factory import AgentFactory
from covenant.agents.communication_layer import CommunicationLayer

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Status of an agent in the swarm."""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    FAILED = "failed"
    SUSPICIOUS = "suspicious"

class AgentRole(Enum):
    """Roles an agent can have in the swarm."""
    WORKER = "worker"
    VALIDATOR = "validator"
    COORDINATOR = "coordinator"
    MONITOR = "monitor"
    BACKUP = "backup"

@dataclass
class AgentInfo:
    """Information about an agent in the swarm."""
    agent_id: str
    role: AgentRole
    capabilities: List[str]
    status: AgentStatus
    performance_score: float = 1.0
    trust_score: float = 1.0
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Task:
    """A task to be executed by the swarm."""
    task_id: str
    task_type: str
    priority: int
    parameters: Dict[str, Any]
    required_capabilities: List[str]
    timeout: float = 60.0
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    status: str = "pending"
    assigned_agent: Optional[str] = None

class SwarmOrchestrator:
    """
    Orchestrates a swarm of AI agents with Byzantine fault tolerance.
    """
    
    def __init__(
        self,
        swarm_id: str,
        consensus_protocol: Optional[ConsensusProtocol] = None,
        max_agents: int = 100,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the swarm orchestrator.
        
        Args:
            swarm_id: Unique identifier for the swarm
            consensus_protocol: Consensus protocol instance
            max_agents: Maximum number of agents in the swarm
            config: Configuration dictionary
        """
        self.swarm_id = swarm_id
        self.config = config or {}
        self.max_agents = max_agents
        
        # Agent management
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_factory = AgentFactory()
        self.communication = CommunicationLayer(swarm_id)
        
        # Consensus protocol
        self.consensus = consensus_protocol or ConsensusProtocol()
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.completed_tasks: Dict[str, Task] = {}
        
        # Swarm state
        self.epoch = 0
        self.leader: Optional[str] = None
        self.is_running = False
        
        # Performance metrics
        self.metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_completion_time': 0.0,
            'agent_utilization': 0.0,
            'consensus_rounds': 0,
            'byzantine_detections': 0
        }
        
        # Heartbeat monitoring
        self.heartbeat_interval = self.config.get('heartbeat_interval', 5.0)
        self.heartbeat_timeout = self.config.get('heartbeat_timeout', 15.0)
        
        logger.info(f"SwarmOrchestrator initialized with ID: {swarm_id}")
    
    async def start(self):
        """Start the swarm orchestrator."""
        self.is_running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._heartbeat_monitor()),
            asyncio.create_task(self._task_dispatcher()),
            asyncio.create_task(self._consensus_monitor()),
            asyncio.create_task(self._performance_optimizer())
        ]
        
        logger.info(f"Swarm {self.swarm_id} started")
    
    async def stop(self):
        """Stop the swarm orchestrator."""
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Stop all agents
        for agent_id in list(self.agents.keys()):
            await self.remove_agent(agent_id)
        
        logger.info(f"Swarm {self.swarm_id} stopped")
    
    async def add_agent(
        self,
        role: AgentRole = AgentRole.WORKER,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new agent to the swarm."""
        if len(self.agents) >= self.max_agents:
            raise RuntimeError(f"Swarm has reached maximum capacity: {self.max_agents}")
        
        # Generate agent ID
        agent_id = self._generate_agent_id(role, capabilities)
        
        # Create agent instance
        agent = self.agent_factory.create_agent(
            agent_id=agent_id,
            role=role,
            capabilities=capabilities or [],
            swarm_id=self.swarm_id
        )
        
        # Register agent
        self.agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            role=role,
            capabilities=capabilities or [],
            status=AgentStatus.IDLE,
            metadata=metadata or {}
        )
        
        # Initialize agent
        await agent.initialize()
        
        # Update leader if needed
        if role == AgentRole.COORDINATOR and not self.leader:
            self.leader = agent_id
        
        logger.info(f"Added agent {agent_id} with role {role.value}")
        return agent_id
    
    async def remove_agent(self, agent_id: str):
        """Remove an agent from the swarm."""
        if agent_id in self.agents:
            agent_info = self.agents[agent_id]
            
            # Reassign tasks if agent was busy
            if agent_info.status == AgentStatus.BUSY:
                for task in self.tasks.values():
                    if task.assigned_agent == agent_id:
                        task.status = "pending"
                        task.assigned_agent = None
                        await self.submit_task(task)
            
            # Remove agent
            del self.agents[agent_id]
            
            # Update leader if needed
            if agent_id == self.leader:
                self.leader = await self._elect_leader()
            
            logger.info(f"Removed agent {agent_id}")
        else:
            logger.warning(f"Agent {agent_id} not found")
    
    async def submit_task(self, task: Task) -> str:
        """Submit a task to the swarm."""
        # Generate task ID if not provided
        if not task.task_id:
            task.task_id = self._generate_task_id(task)
        
        # Add to task queue and registry
        self.tasks[task.task_id] = task
        self.metrics['total_tasks'] += 1
        
        # Calculate priority (higher number = higher priority)
        priority = -task.priority  # Negative because PriorityQueue is min-heap
        await self.task_queue.put((priority, task.task_id))
        
        logger.info(f"Submitted task {task.task_id} with priority {task.priority}")
        return task.task_id
    
    async def get_task_result(self, task_id: str, timeout: float = 30.0) -> Any:
        """Get the result of a completed task."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                return task.result
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Timeout waiting for task {task_id}")
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats and detect failures."""
        while self.is_running:
            try:
                current_time = time.time()
                failed_agents = []
                suspicious_agents = []
                
                # Check each agent's heartbeat
                for agent_id, agent_info in self.agents.items():
                    time_since_heartbeat = current_time - agent_info.last_heartbeat
                    
                    if time_since_heartbeat > self.heartbeat_timeout:
                        # Agent has failed
                        agent_info.status = AgentStatus.FAILED
                        failed_agents.append(agent_id)
                        logger.warning(f"Agent {agent_id} failed (no heartbeat)")
                    
                    elif time_since_heartbeat > self.heartbeat_timeout * 0.7:
                        # Agent is suspicious
                        if agent_info.status != AgentStatus.SUSPICIOUS:
                            agent_info.status = AgentStatus.SUSPICIOUS
                            suspicious_agents.append(agent_id)
                            logger.warning(f"Agent {agent_id} is suspicious (slow heartbeat)")
                
                # Handle failed agents
                for agent_id in failed_agents:
                    await self.remove_agent(agent_id)
                
                # Handle suspicious agents
                for agent_id in suspicious_agents:
                    await self._investigate_agent(agent_id)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _task_dispatcher(self):
        """Dispatch tasks to available agents."""
        while self.is_running:
            try:
                # Get next task from queue
                priority, task_id = await self.task_queue.get()
                
                if task_id not in self.tasks:
                    continue
                
                task = self.tasks[task_id]
                
                # Check dependencies
                if task.dependencies:
                    dependencies_met = all(
                        dep_id in self.completed_tasks
                        for dep_id in task.dependencies
                    )
                    
                    if not dependencies_met:
                        # Requeue task
                        await self.task_queue.put((priority, task_id))
                        await asyncio.sleep(1)
                        continue
                
                # Find suitable agent
                suitable_agents = await self._find_suitable_agents(task)
                
                if suitable_agents:
                    # Select best agent
                    agent_id = await self._select_best_agent(suitable_agents, task)
                    
                    if agent_id:
                        # Assign task to agent
                        task.assigned_agent = agent_id
                        task.status = "assigned"
                        
                        agent_info = self.agents[agent_id]
                        agent_info.status = AgentStatus.BUSY
                        
                        # Execute task asynchronously
                        asyncio.create_task(self._execute_task(task, agent_id))
                    else:
                        # No suitable agent found, requeue
                        await self.task_queue.put((priority, task_id))
                        await asyncio.sleep(1)
                else:
                    # No suitable agents, requeue
                    await self.task_queue.put((priority, task_id))
                    await asyncio.sleep(5)
                
                # Mark task as processed
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task dispatcher error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: Task, agent_id: str):
        """Execute a task with an agent."""
        try:
            # Get agent instance
            agent = self.agent_factory.get_agent(agent_id)
            
            if not agent:
                raise RuntimeError(f"Agent {agent_id} not found")
            
            # Execute task
            start_time = time.time()
            result = await agent.execute_task(task)
            
            # Validate result with consensus
            if task.task_type in ["critical", "sensitive"]:
                validation_result = await self.consensus.validate_result(
                    task=task,
                    result=result,
                    proposer_id=agent_id,
                    swarm_agents=list(self.agents.values())
                )
                
                if not validation_result.valid:
                    logger.warning(f"Task {task.task_id} result rejected by consensus")
                    result = None
            
            completion_time = time.time() - start_time
            
            # Update task
            task.result = result
            task.status = "completed" if result else "failed"
            
            # Update agent info
            agent_info = self.agents[agent_id]
            agent_info.status = AgentStatus.IDLE
            agent_info.last_heartbeat = time.time()
            
            # Update performance metrics
            self._update_task_metrics(task, completion_time, result is not None)
            
            # Move to completed tasks
            self.completed_tasks[task.task_id] = task
            if task.task_id in self.tasks:
                del self.tasks[task.task_id]
            
            logger.info(f"Task {task.task_id} completed by agent {agent_id} in {completion_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Task execution error for task {task.task_id}: {e}")
            
            # Mark task as failed
            task.status = "failed"
            task.result = None
            
            # Update agent status
            if agent_id in self.agents:
                agent_info = self.agents[agent_id]
                agent_info.status = AgentStatus.IDLE
                agent_info.performance_score *= 0.9  # Penalize performance
            
            self.metrics['failed_tasks'] += 1
    
    async def _find_suitable_agents(self, task: Task) -> List[str]:
        """Find agents suitable for a task."""
        suitable_agents = []
        
        for agent_id, agent_info in self.agents.items():
            # Check if agent is available
            if agent_info.status != AgentStatus.IDLE:
                continue
            
            # Check capabilities
            if not self._has_capabilities(agent_info.capabilities, task.required_capabilities):
                continue
            
            # Check trust score
            if agent_info.trust_score < 0.3:
                continue
            
            suitable_agents.append(agent_id)
        
        return suitable_agents
    
    async def _select_best_agent(self, agent_ids: List[str], task: Task) -> Optional[str]:
        """Select the best agent for a task."""
        if not agent_ids:
            return None
        
        # Calculate scores for each agent
        scores = {}
        for agent_id in agent_ids:
            agent_info = self.agents[agent_id]
            
            # Base score
            score = agent_info.performance_score * agent_info.trust_score
            
            # Adjust for role
            if agent_info.role == AgentRole.COORDINATOR:
                score *= 1.2
            elif agent_info.role == AgentRole.VALIDATOR:
                score *= 1.1
            
            # Adjust for load balancing (prefer less busy agents)
            busy_count = sum(
                1 for t in self.tasks.values()
                if t.assigned_agent == agent_id
            )
            score *= (1.0 / (busy_count + 1))
            
            scores[agent_id] = score
        
        # Select agent with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    async def _investigate_agent(self, agent_id: str):
        """Investigate a suspicious agent."""
        if agent_id not in self.agents:
            return
        
        agent_info = self.agents[agent_id]
        
        # Perform health check
        health_check = await self._perform_health_check(agent_id)
        
        if health_check['healthy']:
            # Agent is healthy, restore status
            agent_info.status = AgentStatus.IDLE
            agent_info.trust_score *= 0.95  # Slight penalty for being suspicious
            logger.info(f"Agent {agent_id} cleared suspicion")
        else:
            # Agent is unhealthy, mark as failed
            agent_info.status = AgentStatus.FAILED
            await self.remove_agent(agent_id)
    
    async def _perform_health_check(self, agent_id: str) -> Dict[str, Any]:
        """Perform health check on an agent."""
        try:
            # Get agent instance
            agent = self.agent_factory.get_agent(agent_id)
            
            if not agent:
                return {'healthy': False, 'reason': 'Agent not found'}
            
            # Simple health check task
            health_task = Task(
                task_id=f"health_check_{int(time.time())}",
                task_type="health_check",
                priority=100,
                parameters={'test': 'ping'},
                required_capabilities=[],
                timeout=5.0
            )
            
            # Execute health check
            result = await agent.execute_task(health_task)
            
            return {
                'healthy': result is not None,
                'response_time': result.get('response_time', 0) if isinstance(result, dict) else 0,
                'reason': 'Health check passed' if result else 'No response'
            }
            
        except Exception as e:
            return {'healthy': False, 'reason': str(e)}
    
    async def _consensus_monitor(self):
        """Monitor and participate in consensus protocol."""
        while self.is_running:
            try:
                # Participate in consensus rounds
                if self.agents:
                    consensus_result = await self.consensus.run_round(
                        epoch=self.epoch,
                        participants=list(self.agents.values())
                    )
                    
                    self.metrics['consensus_rounds'] += 1
                    
                    if consensus_result.byzantine_detected:
                        self.metrics['byzantine_detections'] += 1
                        
                        # Handle Byzantine agents
                        for agent_id in consensus_result.byzantine_agents:
                            if agent_id in self.agents:
                                logger.warning(f"Byzantine agent detected: {agent_id}")
                                await self.remove_agent(agent_id)
                
                await asyncio.sleep(self.consensus.round_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consensus monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _performance_optimizer(self):
        """Optimize swarm performance through learning."""
        while self.is_running:
            try:
                # Update agent performance scores based on task completion
                for task in self.completed_tasks.values():
                    if task.assigned_agent and task.assigned_agent in self.agents:
                        agent_info = self.agents[task.assigned_agent]
                        
                        if task.status == "completed":
                            # Reward successful completion
                            agent_info.performance_score = min(
                                1.0,
                                agent_info.performance_score * 1.01
                            )
                            agent_info.trust_score = min(
                                1.0,
                                agent_info.trust_score * 1.005
                            )
                        else:
                            # Penalize failure
                            agent_info.performance_score *= 0.95
                            agent_info.trust_score *= 0.9
                
                # Periodically clean up old completed tasks
                current_time = time.time()
                old_tasks = [
                    task_id for task_id, task in self.completed_tasks.items()
                    if current_time - getattr(task, 'completion_time', current_time) > 3600
                ]
                
                for task_id in old_tasks:
                    del self.completed_tasks[task_id]
                
                await asyncio.sleep(60)  # Run every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance optimizer error: {e}")
                await asyncio.sleep(60)
    
    async def _elect_leader(self) -> Optional[str]:
        """Elect a new leader for the swarm."""
        if not self.agents:
            return None
        
        # Filter for coordinator-capable agents
        coordinator_candidates = [
            (agent_id, agent_info)
            for agent_id, agent_info in self.agents.items()
            if agent_info.role in [AgentRole.COORDINATOR, AgentRole.VALIDATOR]
            and agent_info.status == AgentStatus.IDLE
            and agent_info.trust_score > 0.7
        ]
        
        if not coordinator_candidates:
            # Fall back to any agent with high trust
            coordinator_candidates = [
                (agent_id, agent_info)
                for agent_id, agent_info in self.agents.items()
                if agent_info.status == AgentStatus.IDLE
                and agent_info.trust_score > 0.7
            ]
        
        if not coordinator_candidates:
            return None
        
        # Select agent with highest trust score
        best_agent_id, best_agent_info = max(
            coordinator_candidates,
            key=lambda x: x[1].trust_score
        )
        
        return best_agent_id
    
    def _has_capabilities(self, agent_capabilities: List[str], required_capabilities: List[str]) -> bool:
        """Check if agent has all required capabilities."""
        return all(cap in agent_capabilities for cap in required_capabilities)
    
    def _generate_agent_id(self, role: AgentRole, capabilities: Optional[List[str]]) -> str:
        """Generate a unique agent ID."""
        cap_str = ''.join(sorted(capabilities or []))
        seed = f"{role.value}_{cap_str}_{time.time()}_{random.random()}"
        return hashlib.sha256(seed.encode()).hexdigest()[:16]
    
    def _generate_task_id(self, task: Task) -> str:
        """Generate a unique task ID."""
        task_str = json.dumps({
            'type': task.task_type,
            'params': task.parameters,
            'timestamp': time.time()
        }, sort_keys=True)
        return hashlib.sha256(task_str.encode()).hexdigest()[:16]
    
    def _update_task_metrics(self, task: Task, completion_time: float, success: bool):
        """Update metrics after task completion."""
        if success:
            self.metrics['completed_tasks'] += 1
            
            # Update average completion time
            old_avg = self.metrics['average_completion_time']
            n = self.metrics['completed_tasks']
            self.metrics['average_completion_time'] = (old_avg * (n - 1) + completion_time) / n
        else:
            self.metrics['failed_tasks'] += 1
        
        # Update agent utilization
        busy_agents = sum(
            1 for agent in self.agents.values()
            if agent.status == AgentStatus.BUSY
        )
        total_agents = len(self.agents)
        
        if total_agents > 0:
            self.metrics['agent_utilization'] = busy_agents / total_agents
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current status of the swarm."""
        status_counts = {}
        role_counts = {}
        
        for agent in self.agents.values():
            status_counts[agent.status.value] = status_counts.get(agent.status.value, 0) + 1
            role_counts[agent.role.value] = role_counts.get(agent.role.value, 0) + 1
        
        return {
            'swarm_id': self.swarm_id,
            'epoch': self.epoch,
            'total_agents': len(self.agents),
            'leader': self.leader,
            'status_counts': status_counts,
            'role_counts': role_counts,
            'pending_tasks': self.task_queue.qsize(),
            'active_tasks': len([t for t in self.tasks.values() if t.status == "assigned"]),
            'completed_tasks': len(self.completed_tasks),
            'metrics': self.metrics.copy()
        }
    
    def get_agent_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance data for all agents."""
        performance = {}
        
        for agent_id, agent_info in self.agents.items():
            performance[agent_id] = {
                'role': agent_info.role.value,
                'status': agent_info.status.value,
                'performance_score': agent_info.performance_score,
                'trust_score': agent_info.trust_score,
                'capabilities': agent_info.capabilities,
                'last_heartbeat': agent_info.last_heartbeat,
                'is_healthy': time.time() - agent_info.last_heartbeat < self.heartbeat_timeout
            }
        
        return performance
