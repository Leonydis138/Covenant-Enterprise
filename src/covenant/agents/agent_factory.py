"""
Agent Factory for creating and managing AI agents.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_type: str
    capabilities: List[str]
    max_tasks: int = 10
    priority: int = 1
    metadata: Dict[str, Any] = None

class AgentFactory:
    """
    Factory for creating and initializing AI agents.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent factory.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.agent_templates = {}
        self.created_agents = {}
        
        # Register default agent types
        self._register_default_types()
        
        logger.info("AgentFactory initialized")
    
    def _register_default_types(self):
        """Register default agent types."""
        self.agent_templates['worker'] = AgentConfig(
            agent_type='worker',
            capabilities=['execute', 'compute'],
            max_tasks=5
        )
        
        self.agent_templates['validator'] = AgentConfig(
            agent_type='validator',
            capabilities=['validate', 'verify'],
            max_tasks=10
        )
        
        self.agent_templates['coordinator'] = AgentConfig(
            agent_type='coordinator',
            capabilities=['coordinate', 'plan', 'delegate'],
            max_tasks=20
        )
        
        self.agent_templates['monitor'] = AgentConfig(
            agent_type='monitor',
            capabilities=['monitor', 'audit', 'alert'],
            max_tasks=100
        )
    
    def create_agent(
        self,
        agent_type: str,
        agent_id: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new agent instance.
        
        Args:
            agent_type: Type of agent to create
            agent_id: Optional custom agent ID
            custom_config: Optional custom configuration
            
        Returns:
            Agent instance dictionary
        """
        # Generate agent ID if not provided
        if not agent_id:
            agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
        
        # Get template
        if agent_type not in self.agent_templates:
            logger.warning(f"Unknown agent type: {agent_type}, using worker template")
            template = self.agent_templates['worker']
        else:
            template = self.agent_templates[agent_type]
        
        # Create agent
        agent = {
            'id': agent_id,
            'type': agent_type,
            'capabilities': template.capabilities.copy(),
            'max_tasks': template.max_tasks,
            'priority': template.priority,
            'status': 'idle',
            'current_task': None,
            'task_history': [],
            'performance_metrics': {
                'tasks_completed': 0,
                'tasks_failed': 0,
                'average_completion_time': 0.0
            },
            'created_at': None,
            'metadata': template.metadata or {}
        }
        
        # Apply custom configuration
        if custom_config:
            agent.update(custom_config)
        
        # Store agent
        self.created_agents[agent_id] = agent
        
        logger.info(f"Created agent: {agent_id} ({agent_type})")
        return agent
    
    def create_agents(
        self,
        agent_specs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create multiple agents at once.
        
        Args:
            agent_specs: List of agent specifications
            
        Returns:
            List of created agent instances
        """
        agents = []
        for spec in agent_specs:
            agent_type = spec.get('type', 'worker')
            agent_id = spec.get('id')
            config = spec.get('config')
            
            agent = self.create_agent(agent_type, agent_id, config)
            agents.append(agent)
        
        return agents
    
    def register_agent_type(
        self,
        agent_type: str,
        config: AgentConfig
    ):
        """
        Register a new agent type template.
        
        Args:
            agent_type: Name of the agent type
            config: Configuration for the agent type
        """
        self.agent_templates[agent_type] = config
        logger.info(f"Registered agent type: {agent_type}")
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent by ID."""
        return self.created_agents.get(agent_id)
    
    def destroy_agent(self, agent_id: str) -> bool:
        """
        Destroy an agent.
        
        Args:
            agent_id: ID of the agent to destroy
            
        Returns:
            True if agent was destroyed
        """
        if agent_id in self.created_agents:
            del self.created_agents[agent_id]
            logger.info(f"Destroyed agent: {agent_id}")
            return True
        return False
    
    def get_agent_count(self) -> int:
        """Get the total number of created agents."""
        return len(self.created_agents)
    
    def get_agents_by_type(self, agent_type: str) -> List[Dict[str, Any]]:
        """Get all agents of a specific type."""
        return [
            agent for agent in self.created_agents.values()
            if agent['type'] == agent_type
        ]
    
    def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get all agents that are currently idle."""
        return [
            agent for agent in self.created_agents.values()
            if agent['status'] == 'idle'
        ]
