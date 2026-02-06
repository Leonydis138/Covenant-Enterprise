"""
Agent Factory for creating and managing AI agents (object-oriented).
"""

import logging
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import asyncio

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# -------------------------
# Agent Configuration
# -------------------------
@dataclass
class AgentConfig:
    """Template for an agent type."""
    agent_type: str
    capabilities: List[str]
    max_tasks: int = 10
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

# -------------------------
# Base Agent Class
# -------------------------
class BaseAgent:
    """
    Base AI agent object with state, tasks, and performance tracking.
    """
    def __init__(self, agent_id: str, agent_type: str, capabilities: List[str], config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.max_tasks = config.get("max_tasks", 10) if config else 10
        self.priority = config.get("priority", 1) if config else 1
        self.metadata = config.get("metadata", {}) if config else {}
        self.status = "idle"
        self.current_task = None
        self.task_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_completion_time": 0.0
        }

    async def initialize(self):
        """Simulate initialization."""
        self.status = "ready"
        logger.debug(f"Agent {self.agent_id} initialized")
        await asyncio.sleep(0)  # placeholder for async init
        return True

    async def execute_task(self, task: Dict[str, Any]):
        """Execute a task asynchronously."""
        if self.status != "ready" and self.status != "idle":
            logger.warning(f"Agent {self.agent_id} busy")
            return {"status": "failed", "reason": "agent busy"}

        self.status = "busy"
        self.current_task = task
        logger.info(f"Agent {self.agent_id} executing task {task.get('id')}")
        
        try:
            # Simulate task execution (override for real logic)
            await asyncio.sleep(task.get("duration", 0.1))
            result = {"status": "success", "result": "task_completed"}
            
            # Update metrics
            self.performance_metrics["tasks_completed"] += 1
            self.task_history.append(task)
            return result
        except Exception as e:
            self.performance_metrics["tasks_failed"] += 1
            return {"status": "failed", "reason": str(e)}
        finally:
            self.current_task = None
            self.status = "idle"

# -------------------------
# Agent Factory
# -------------------------
class AgentFactory:
    """
    Factory for creating and managing AI agents.
    """
    def __init__(self):
        self.agent_templates: Dict[str, AgentConfig] = {}
        self.created_agents: Dict[str, BaseAgent] = {}
        self._register_default_types()
        logger.info("AgentFactory initialized")

    # -------------------------
    # Default Agent Types
    # -------------------------
    def _register_default_types(self):
        self.register_agent_type("worker", AgentConfig(agent_type="worker", capabilities=["execute", "compute"], max_tasks=5))
        self.register_agent_type("validator", AgentConfig(agent_type="validator", capabilities=["validate", "verify"], max_tasks=10))
        self.register_agent_type("coordinator", AgentConfig(agent_type="coordinator", capabilities=["coordinate", "plan", "delegate"], max_tasks=20))
        self.register_agent_type("monitor", AgentConfig(agent_type="monitor", capabilities=["monitor", "audit", "alert"], max_tasks=100))

    # -------------------------
    # Create Agents
    # -------------------------
    def create_agent(self, agent_type: str, agent_id: Optional[str] = None, custom_config: Optional[Dict[str, Any]] = None) -> BaseAgent:
        if not agent_id:
            agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
        
        template = self.agent_templates.get(agent_type, self.agent_templates["worker"])
        agent = BaseAgent(agent_id, agent_type, template.capabilities.copy(), custom_config or template.__dict__)
        self.created_agents[agent_id] = agent
        logger.info(f"Created agent {agent_id} ({agent_type})")
        return agent

    def create_agents(self, agent_specs: List[Dict[str, Any]]) -> List[BaseAgent]:
        agents = []
        for spec in agent_specs:
            agent_type = spec.get("type", "worker")
            agent_id = spec.get("id")
            config = spec.get("config")
            agent = self.create_agent(agent_type, agent_id, config)
            agents.append(agent)
        return agents

    # -------------------------
    # Agent Type Management
    # -------------------------
    def register_agent_type(self, agent_type: str, config: AgentConfig):
        self.agent_templates[agent_type] = config
        logger.info(f"Registered agent type {agent_type}")

    # -------------------------
    # Query / Management
    # -------------------------
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        return self.created_agents.get(agent_id)

    def destroy_agent(self, agent_id: str) -> bool:
        if agent_id in self.created_agents:
            del self.created_agents[agent_id]
            logger.info(f"Destroyed agent {agent_id}")
            return True
        return False

    def get_agent_count(self) -> int:
        return len(self.created_agents)

    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        return [a for a in self.created_agents.values() if a.agent_type == agent_type]

    def get_available_agents(self) -> List[BaseAgent]:
        return [a for a in self.created_agents.values() if a.status == "idle"]
