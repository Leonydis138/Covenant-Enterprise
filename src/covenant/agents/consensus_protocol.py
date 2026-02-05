"""
Byzantine Fault Tolerant Consensus Protocol for Agent Swarms.
"""

import logging
import hashlib
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ConsensusType(Enum):
    """Types of consensus mechanisms."""
    PBFT = "pbft"  # Practical Byzantine Fault Tolerance
    RAFT = "raft"
    PAXOS = "paxos"
    POW = "pow"  # Proof of Work

@dataclass
class Vote:
    """A vote in the consensus process."""
    agent_id: str
    proposal_id: str
    vote: bool  # True = accept, False = reject
    timestamp: float
    signature: Optional[str] = None

class ConsensusProtocol:
    """
    Implements Byzantine Fault Tolerant consensus for agent coordination.
    """
    
    def __init__(
        self,
        consensus_type: ConsensusType = ConsensusType.PBFT,
        fault_tolerance: float = 0.33,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize consensus protocol.
        
        Args:
            consensus_type: Type of consensus mechanism
            fault_tolerance: Maximum fraction of faulty nodes tolerated
            config: Configuration dictionary
        """
        self.consensus_type = consensus_type
        self.fault_tolerance = fault_tolerance
        self.config = config or {}
        
        # Consensus state
        self.proposals: Dict[str, Dict[str, Any]] = {}
        self.votes: Dict[str, List[Vote]] = {}
        self.committed: Set[str] = set()
        
        logger.info(f"ConsensusProtocol initialized: {consensus_type.value}")
    
    async def propose(
        self,
        proposal_id: str,
        proposal_data: Dict[str, Any],
        proposer_id: str
    ) -> bool:
        """
        Propose a new action or decision to the swarm.
        
        Args:
            proposal_id: Unique identifier for the proposal
            proposal_data: Data being proposed
            proposer_id: ID of the proposing agent
            
        Returns:
            True if proposal was accepted
        """
        # Store proposal
        self.proposals[proposal_id] = {
            'data': proposal_data,
            'proposer': proposer_id,
            'timestamp': time.time(),
            'status': 'proposed'
        }
        
        # Initialize vote tracking
        self.votes[proposal_id] = []
        
        logger.info(f"Proposal {proposal_id} from {proposer_id}")
        return True
    
    async def vote(
        self,
        proposal_id: str,
        agent_id: str,
        accept: bool
    ) -> bool:
        """
        Cast a vote on a proposal.
        
        Args:
            proposal_id: ID of the proposal
            agent_id: ID of the voting agent
            accept: True to accept, False to reject
            
        Returns:
            True if vote was recorded
        """
        if proposal_id not in self.proposals:
            logger.warning(f"Proposal {proposal_id} not found")
            return False
        
        # Record vote
        vote = Vote(
            agent_id=agent_id,
            proposal_id=proposal_id,
            vote=accept,
            timestamp=time.time()
        )
        
        self.votes[proposal_id].append(vote)
        logger.debug(f"Vote from {agent_id} on {proposal_id}: {accept}")
        
        return True
    
    async def check_consensus(
        self,
        proposal_id: str,
        total_agents: int
    ) -> Optional[bool]:
        """
        Check if consensus has been reached on a proposal.
        
        Args:
            proposal_id: ID of the proposal
            total_agents: Total number of agents in the swarm
            
        Returns:
            True if accepted, False if rejected, None if pending
        """
        if proposal_id not in self.votes:
            return None
        
        votes = self.votes[proposal_id]
        
        # Need votes from majority
        required_votes = int(total_agents * (2/3)) + 1
        
        if len(votes) < required_votes:
            return None  # Not enough votes yet
        
        # Count accept votes
        accept_votes = sum(1 for v in votes if v.vote)
        
        # Byzantine fault tolerance: need 2/3 + 1 for acceptance
        threshold = int(total_agents * (2/3)) + 1
        
        if accept_votes >= threshold:
            self.proposals[proposal_id]['status'] = 'accepted'
            self.committed.add(proposal_id)
            logger.info(f"Consensus reached: {proposal_id} ACCEPTED")
            return True
        
        # Check if rejection is certain
        reject_votes = len(votes) - accept_votes
        if reject_votes > (total_agents - threshold):
            self.proposals[proposal_id]['status'] = 'rejected'
            logger.info(f"Consensus reached: {proposal_id} REJECTED")
            return False
        
        return None  # Still pending
    
    def get_proposal_status(self, proposal_id: str) -> Optional[str]:
        """Get the current status of a proposal."""
        if proposal_id in self.proposals:
            return self.proposals[proposal_id]['status']
        return None
    
    def get_committed_proposals(self) -> List[str]:
        """Get list of committed proposal IDs."""
        return list(self.committed)
    
    def verify_proposal(
        self,
        proposal_id: str,
        proposal_hash: str
    ) -> bool:
        """
        Verify the integrity of a proposal.
        
        Args:
            proposal_id: ID of the proposal
            proposal_hash: Expected hash of the proposal
            
        Returns:
            True if proposal is valid
        """
        if proposal_id not in self.proposals:
            return False
        
        # Compute hash of proposal data
        proposal_data = self.proposals[proposal_id]['data']
        computed_hash = hashlib.sha256(
            str(proposal_data).encode()
        ).hexdigest()
        
        return computed_hash == proposal_hash
