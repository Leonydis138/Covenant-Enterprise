"""
Blockchain-based immutable audit trail for constitutional AI decisions.
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import asyncio

from web3 import Web3
from eth_account import Account
import ipfshttpclient

from covenant.blockchain.zk_proofs import ZeroKnowledgeProofs
from covenant.blockchain.smart_contract import SmartContract

logger = logging.getLogger(__name__)

@dataclass
class AuditEntry:
    """An entry in the audit trail."""
    entry_id: str
    timestamp: datetime
    action_id: str
    agent_id: str
    decision: str
    reason: str
    evidence_hash: str
    previous_hash: str
    block_number: Optional[int] = None
    transaction_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MerkleProof:
    """Merkle proof for audit trail verification."""
    leaf_hash: str
    path: List[Tuple[str, bool]]  # (hash, is_left)
    root_hash: str
    tree_depth: int

class BlockchainAuditTrail:
    """
    Immutable audit trail using blockchain technology.
    """
    
    def __init__(
        self,
        blockchain_config: Optional[Dict[str, Any]] = None,
        use_ipfs: bool = True
    ):
        """
        Initialize the blockchain audit trail.
        
        Args:
            blockchain_config: Blockchain configuration
            use_ipfs: Whether to use IPFS for evidence storage
        """
        self.config = blockchain_config or {}
        self.use_ipfs = use_ipfs
        
        # Initialize blockchain connection
        self.web3 = None
        self.contract = None
        self.account = None
        
        # Initialize IPFS client
        self.ipfs_client = None
        
        # Initialize zero-knowledge proofs
        self.zk_proofs = ZeroKnowledgeProofs()
        
        # Local audit trail (for verification and caching)
        self.audit_chain: List[AuditEntry] = []
        self.merkle_tree = {}
        self.root_hash = ""
        
        # Performance metrics
        self.metrics = {
            'total_entries': 0,
            'blockchain_entries': 0,
            'ipfs_uploads': 0,
            'verifications': 0,
            'average_confirmation_time': 0.0
        }
        
        # Initialize connections
        self._initialize_connections()
        
        logger.info("BlockchainAuditTrail initialized")
    
    def _initialize_connections(self):
        """Initialize blockchain and IPFS connections."""
        # Initialize Web3 connection
        try:
            if 'provider_url' in self.config:
                self.web3 = Web3(Web3.HTTPProvider(self.config['provider_url']))
                
                if self.web3.is_connected():
                    logger.info(f"Connected to blockchain at {self.config['provider_url']}")
                    
                    # Load account
                    if 'private_key' in self.config:
                        self.account = Account.from_key(self.config['private_key'])
                        logger.info(f"Loaded account: {self.account.address}")
                    
                    # Load smart contract
                    if 'contract_address' in self.config and 'contract_abi' in self.config:
                        self.contract = self.web3.eth.contract(
                            address=self.config['contract_address'],
                            abi=self.config['contract_abi']
                        )
                        logger.info(f"Loaded contract at {self.config['contract_address']}")
                else:
                    logger.warning("Failed to connect to blockchain")
            else:
                logger.info("No blockchain provider configured, using local audit trail only")
        except Exception as e:
            logger.error(f"Failed to initialize blockchain connection: {e}")
        
        # Initialize IPFS client
        if self.use_ipfs:
            try:
                self.ipfs_client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
                logger.info("Connected to IPFS daemon")
            except Exception as e:
                logger.warning(f"Failed to connect to IPFS: {e}")
                self.ipfs_client = None
    
    async def log_decision(
        self,
        action_id: str,
        agent_id: str,
        decision: str,
        reason: str,
        evidence: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEntry:
        """
        Log a constitutional decision to the audit trail.
        
        Args:
            action_id: ID of the action being evaluated
            agent_id: ID of the agent making the decision
            decision: The decision (allow/block/modify)
            reason: Reason for the decision
            evidence: Supporting evidence for the decision
            metadata: Additional metadata
            
        Returns:
            Audit entry with blockchain/IPFS references
        """
        start_time = time.time()
        
        try:
            # Generate evidence hash
            evidence_hash = self._hash_evidence(evidence)
            
            # Store evidence on IPFS if available
            ipfs_cid = None
            if self.ipfs_client and evidence:
                ipfs_cid = await self._store_on_ipfs(evidence)
                self.metrics['ipfs_uploads'] += 1
            
            # Create audit entry
            previous_hash = self.root_hash if self.audit_chain else "0" * 64
            timestamp = datetime.utcnow()
            
            entry = AuditEntry(
                entry_id=self._generate_entry_id(action_id, agent_id, timestamp),
                timestamp=timestamp,
                action_id=action_id,
                agent_id=agent_id,
                decision=decision,
                reason=reason,
                evidence_hash=evidence_hash,
                previous_hash=previous_hash,
                metadata={
                    'ipfs_cid': ipfs_cid,
                    **(metadata or {})
                }
            )
            
            # Update local audit chain
            self.audit_chain.append(entry)
            
            # Update Merkle tree
            await self._update_merkle_tree(entry)
            
            # Record on blockchain if available
            if self.contract and self.account:
                try:
                    tx_hash = await self._record_on_blockchain(entry)
                    
                    if tx_hash:
                        entry.transaction_hash = tx_hash
                        
                        # Wait for confirmation
                        block_number = await self._wait_for_confirmation(tx_hash)
                        if block_number:
                            entry.block_number = block_number
                            self.metrics['blockchain_entries'] += 1
                        
                        # Update confirmation time metrics
                        confirmation_time = time.time() - start_time
                        self._update_confirmation_metrics(confirmation_time)
                except Exception as e:
                    logger.error(f"Failed to record on blockchain: {e}")
            
            self.metrics['total_entries'] += 1
            logger.info(f"Logged decision for action {action_id}: {decision}")
            
            return entry
            
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
            raise
    
    async def verify_entry(self, entry: AuditEntry) -> Dict[str, Any]:
        """
        Verify an audit entry's integrity and authenticity.
        
        Args:
            entry: Audit entry to verify
            
        Returns:
            Verification result
        """
        self.metrics['verifications'] += 1
        
        verification_result = {
            'entry_id': entry.entry_id,
            'local_verification': False,
            'blockchain_verification': False,
            'merkle_proof_verification': False,
            'ipfs_verification': False,
            'zk_proof_available': False,
            'overall_valid': False,
            'details': {}
        }
        
        try:
            # 1. Verify local chain integrity
            if entry in self.audit_chain:
                idx = self.audit_chain.index(entry)
                
                # Check previous hash
                if idx == 0:
                    valid_previous = entry.previous_hash == "0" * 64
                else:
                    previous_entry = self.audit_chain[idx - 1]
                    previous_hash = self._calculate_entry_hash(previous_entry)
                    valid_previous = entry.previous_hash == previous_hash
                
                # Check next hash if exists
                valid_next = True
                if idx < len(self.audit_chain) - 1:
                    next_entry = self.audit_chain[idx + 1]
                    next_previous_hash = next_entry.previous_hash
                    current_hash = self._calculate_entry_hash(entry)
                    valid_next = next_previous_hash == current_hash
                
                verification_result['local_verification'] = valid_previous and valid_next
                verification_result['details']['local_chain_valid'] = valid_previous and valid_next
            
            # 2. Verify on blockchain
            if entry.transaction_hash and self.web3:
                try:
                    tx_receipt = self.web3.eth.get_transaction_receipt(entry.transaction_hash)
                    
                    if tx_receipt and tx_receipt.status == 1:
                        # Transaction was successful
                        
                        # Get event logs from contract
                        if self.contract:
                            event_logs = self.contract.events.DecisionLogged().process_receipt(tx_receipt)
                            
                            for event in event_logs:
                                if event.args.entryId == entry.entry_id:
                                    # Verify event data matches entry
                                    matches = (
                                        event.args.actionId == entry.action_id and
                                        event.args.agentId == entry.agent_id and
                                        event.args.decision == entry.decision and
                                        event.args.evidenceHash == entry.evidence_hash
                                    )
                                    
                                    verification_result['blockchain_verification'] = matches
                                    verification_result['details']['blockchain_match'] = matches
                                    break
                except Exception as e:
                    logger.error(f"Blockchain verification error: {e}")
            
            # 3. Verify Merkle proof
            if entry.entry_id in self.merkle_tree:
                proof = self.merkle_tree[entry.entry_id].get('proof')
                if proof:
                    leaf_hash = self._calculate_entry_hash(entry)
                    valid_merkle = self._verify_merkle_proof(leaf_hash, proof, self.root_hash)
                    verification_result['merkle_proof_verification'] = valid_merkle
                    verification_result['details']['merkle_valid'] = valid_merkle
            
            # 4. Verify IPFS evidence
            if entry.metadata.get('ipfs_cid') and self.ipfs_client:
                try:
                    # Retrieve from IPFS
                    evidence = await self._retrieve_from_ipfs(entry.metadata['ipfs_cid'])
                    
                    if evidence:
                        # Calculate hash and compare
                        retrieved_hash = self._hash_evidence(evidence)
                        verification_result['ipfs_verification'] = (
                            retrieved_hash == entry.evidence_hash
                        )
                        verification_result['details']['ipfs_match'] = (
                            retrieved_hash == entry.evidence_hash
                        )
                except Exception as e:
                    logger.error(f"IPFS verification error: {e}")
            
            # 5. Check for zero-knowledge proof
            if 'zk_proof' in entry.metadata:
                verification_result['zk_proof_available'] = True
            
            # Overall validity
            verifications = [
                verification_result['local_verification'],
                verification_result.get('blockchain_verification', True),  # Optional
                verification_result.get('merkle_proof_verification', True),  # Optional
                verification_result.get('ipfs_verification', True)  # Optional
            ]
            
            # Consider valid if at least one verification method succeeded
            verification_result['overall_valid'] = any(verifications)
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            verification_result['details']['error'] = str(e)
            return verification_result
    
    async def generate_zk_proof(
        self,
        entry: AuditEntry,
        reveal_fields: List[str]
    ) -> Dict[str, Any]:
        """
        Generate a zero-knowledge proof for an audit entry.
        
        Args:
            entry: Audit entry
            reveal_fields: Fields to reveal in the proof
            
        Returns:
            Zero-knowledge proof
        """
        try:
            # Create statement to prove
            statement = {
                'entry_id': entry.entry_id,
                'timestamp': entry.timestamp.isoformat(),
                'action_id': entry.action_id,
                'decision': entry.decision,
                'evidence_hash': entry.evidence_hash
            }
            
            # Generate proof
            proof = await self.zk_proofs.generate_proof(
                statement=statement,
                secret_data={
                    'agent_id': entry.agent_id,
                    'reason': entry.reason,
                    'metadata': entry.metadata
                },
                reveal_fields=reveal_fields
            )
            
            # Store proof in entry metadata
            entry.metadata['zk_proof'] = proof
            
            logger.info(f"Generated ZK proof for entry {entry.entry_id}")
            return proof
            
        except Exception as e:
            logger.error(f"Failed to generate ZK proof: {e}")
            raise
    
    async def query_audit_trail(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditEntry]:
        """
        Query the audit trail with filters.
        
        Args:
            filters: Filter criteria
            limit: Maximum number of entries to return
            offset: Offset for pagination
            
        Returns:
            List of audit entries matching the filters
        """
        filtered_entries = self.audit_chain.copy()
        
        if filters:
            for key, value in filters.items():
                if key == 'action_id':
                    filtered_entries = [e for e in filtered_entries if e.action_id == value]
                elif key == 'agent_id':
                    filtered_entries = [e for e in filtered_entries if e.agent_id == value]
                elif key == 'decision':
                    filtered_entries = [e for e in filtered_entries if e.decision == value]
                elif key == 'start_time':
                    filtered_entries = [e for e in filtered_entries if e.timestamp >= value]
                elif key == 'end_time':
                    filtered_entries = [e for e in filtered_entries if e.timestamp <= value]
                elif key == 'has_zk_proof':
                    filtered_entries = [
                        e for e in filtered_entries
                        if 'zk_proof' in e.metadata
                    ]
        
        # Sort by timestamp (newest first)
        filtered_entries.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        
        return filtered_entries[start_idx:end_idx]
    
    async def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the audit trail."""
        if not self.audit_chain:
            return {
                'total_entries': 0,
                'decisions': {},
                'agents': {},
                'time_range': None
            }
        
        # Count decisions
        decisions = {}
        for entry in self.audit_chain:
            decisions[entry.decision] = decisions.get(entry.decision, 0) + 1
        
        # Count agents
        agents = {}
        for entry in self.audit_chain:
            agents[entry.agent_id] = agents.get(entry.agent_id, 0) + 1
        
        # Get time range
        timestamps = [e.timestamp for e in self.audit_chain]
        
        return {
            'total_entries': len(self.audit_chain),
            'decisions': decisions,
            'agents': agents,
            'time_range': {
                'start': min(timestamps).isoformat(),
                'end': max(timestamps).isoformat()
            },
            'blockchain_entries': self.metrics['blockchain_entries'],
            'ipfs_uploads': self.metrics['ipfs_uploads'],
            'root_hash': self.root_hash
        }
    
    def export_audit_trail(self, format: str = 'json') -> str:
        """
        Export the audit trail.
        
        Args:
            format: Export format ('json' or 'csv')
            
        Returns:
            Exported audit trail
        """
        if format == 'json':
            entries_data = []
            for entry in self.audit_chain:
                entry_data = {
                    'entry_id': entry.entry_id,
                    'timestamp': entry.timestamp.isoformat(),
                    'action_id': entry.action_id,
                    'agent_id': entry.agent_id,
                    'decision': entry.decision,
                    'reason': entry.reason,
                    'evidence_hash': entry.evidence_hash,
                    'previous_hash': entry.previous_hash,
                    'block_number': entry.block_number,
                    'transaction_hash': entry.transaction_hash,
                    'metadata': entry.metadata
                }
                entries_data.append(entry_data)
            
            export_data = {
                'audit_trail': entries_data,
                'root_hash': self.root_hash,
                'total_entries': len(self.audit_chain),
                'export_timestamp': datetime.utcnow().isoformat()
            }
            
            return json.dumps(export_data, indent=2)
        
        elif format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                'entry_id', 'timestamp', 'action_id', 'agent_id',
                'decision', 'reason', 'evidence_hash', 'previous_hash',
                'block_number', 'transaction_hash'
            ])
            
            # Write rows
            for entry in self.audit_chain:
                writer.writerow([
                    entry.entry_id,
                    entry.timestamp.isoformat(),
                    entry.action_id,
                    entry.agent_id,
                    entry.decision,
                    entry.reason,
                    entry.evidence_hash,
                    entry.previous_hash,
                    entry.block_number or '',
                    entry.transaction_hash or ''
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def _store_on_ipfs(self, evidence: Dict[str, Any]) -> str:
        """Store evidence on IPFS."""
        if not self.ipfs_client:
            raise RuntimeError("IPFS client not available")
        
        try:
            # Convert evidence to JSON
            evidence_json = json.dumps(evidence, indent=2)
            
            # Add to IPFS
            result = self.ipfs_client.add_bytes(evidence_json.encode())
            cid = result['Hash']
            
            logger.debug(f"Stored evidence on IPFS with CID: {cid}")
            return cid
            
        except Exception as e:
            logger.error(f"Failed to store on IPFS: {e}")
            raise
    
    async def _retrieve_from_ipfs(self, cid: str) -> Optional[Dict[str, Any]]:
        """Retrieve evidence from IPFS."""
        if not self.ipfs_client:
            return None
        
        try:
            # Retrieve from IPFS
            data = self.ipfs_client.cat(cid)
            
            # Parse JSON
            evidence = json.loads(data.decode())
            return evidence
            
        except Exception as e:
            logger.error(f"Failed to retrieve from IPFS: {e}")
            return None
    
    async def _record_on_blockchain(self, entry: AuditEntry) -> Optional[str]:
        """Record audit entry on the blockchain."""
        if not self.contract or not self.account:
            return None
        
        try:
            # Prepare transaction
            nonce = self.web3.eth.get_transaction_count(self.account.address)
            
            # Build transaction
            tx = self.contract.functions.logDecision(
                entry.entry_id,
                entry.action_id,
                entry.agent_id,
                entry.decision,
                entry.reason,
                entry.evidence_hash,
                entry.previous_hash
            ).build_transaction({
                'chainId': self.config.get('chain_id', 1),
                'gas': 200000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': nonce,
                'from': self.account.address
            })
            
            # Sign transaction
            signed_tx = self.account.sign_transaction(tx)
            
            # Send transaction
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            logger.info(f"Submitted transaction {tx_hash.hex()} for entry {entry.entry_id}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Failed to record on blockchain: {e}")
            return None
    
    async def _wait_for_confirmation(self, tx_hash: str, timeout: int = 60) -> Optional[int]:
        """Wait for transaction confirmation."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                receipt = self.web3.eth.get_transaction_receipt(tx_hash)
                if receipt is not None:
                    return receipt.blockNumber
            except:
                pass
            
            await asyncio.sleep(1)
        
        logger.warning(f"Transaction {tx_hash} not confirmed within {timeout} seconds")
        return None
    
    async def _update_merkle_tree(self, entry: AuditEntry):
        """Update Merkle tree with new entry."""
        # Calculate leaf hash
        leaf_hash = self._calculate_entry_hash(entry)
        
        # Store in merkle tree
        if entry.entry_id not in self.merkle_tree:
            self.merkle_tree[entry.entry_id] = {
                'hash': leaf_hash,
                'index': len(self.merkle_tree)
            }
        
        # Recalculate root hash
        self.root_hash = self._calculate_merkle_root()
        
        # Update proofs for all leaves
        await self._update_merkle_proofs()
    
    async def _update_merkle_proofs(self):
        """Update Merkle proofs for all leaves."""
        leaf_count = len(self.merkle_tree)
        
        if leaf_count == 0:
            return
        
        # Get all leaf hashes in order
        leaves = []
        for i in range(leaf_count):
            for entry_id, data in self.merkle_tree.items():
                if data['index'] == i:
                    leaves.append(data['hash'])
                    break
        
        # Calculate proofs for each leaf
        for i, leaf_hash in enumerate(leaves):
            proof = self._calculate_merkle_proof(leaves, i)
            
            # Find entry ID for this leaf
            for entry_id, data in self.merkle_tree.items():
                if data['hash'] == leaf_hash:
                    if 'proof' not in data:
                        data['proof'] = MerkleProof(
                            leaf_hash=leaf_hash,
                            path=proof,
                            root_hash=self.root_hash,
                            tree_depth=len(proof)
                        )
                    else:
                        data['proof'].path = proof
                        data['proof'].root_hash = self.root_hash
                    break
    
    def _calculate_merkle_root(self) -> str:
        """Calculate Merkle root hash."""
        leaf_count = len(self.merkle_tree)
        
        if leaf_count == 0:
            return "0" * 64
        
        # Get all leaf hashes in order
        leaves = []
        for i in range(leaf_count):
            for data in self.merkle_tree.values():
                if data['index'] == i:
                    leaves.append(data['hash'])
                    break
        
        # Calculate Merkle root
        while len(leaves) > 1:
            if len(leaves) % 2 == 1:
                leaves.append(leaves[-1])  # Duplicate last leaf if odd number
            
            new_level = []
            for i in range(0, len(leaves), 2):
                combined = leaves[i] + leaves[i + 1]
                new_hash = hashlib.sha256(combined.encode()).hexdigest()
                new_level.append(new_hash)
            
            leaves = new_level
        
        return leaves[0] if leaves else "0" * 64
    
    def _calculate_merkle_proof(self, leaves: List[str], index: int) -> List[Tuple[str, bool]]:
        """Calculate Merkle proof for a leaf at given index."""
        proof = []
        current_index = index
        
        while len(leaves) > 1:
            if len(leaves) % 2 == 1:
                leaves.append(leaves[-1])
            
            # Check if current index is even or odd
            if current_index % 2 == 0:
                # Current is left child, need right sibling
                sibling_index = current_index + 1
                if sibling_index < len(leaves):
                    proof.append((leaves[sibling_index], False))  # False = right sibling
            else:
                # Current is right child, need left sibling
                sibling_index = current_index - 1
                proof.append((leaves[sibling_index], True))  # True = left sibling
            
            # Move to parent level
            current_index //= 2
            
            # Calculate next level
            new_level = []
            for i in range(0, len(leaves), 2):
                combined = leaves[i] + leaves[i + 1]
                new_hash = hashlib.sha256(combined.encode()).hexdigest()
                new_level.append(new_hash)
            
            leaves = new_level
        
        return proof
    
    def _verify_merkle_proof(
        self,
        leaf_hash: str,
        proof: MerkleProof,
        root_hash: str
    ) -> bool:
        """Verify a Merkle proof."""
        current_hash = leaf_hash
        
        for sibling_hash, is_left in proof.path:
            if is_left:
                # Current is right child
                combined = sibling_hash + current_hash
            else:
                # Current is left child
                combined = current_hash + sibling_hash
            
            current_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        return current_hash == root_hash
    
    def _hash_evidence(self, evidence: Dict[str, Any]) -> str:
        """Calculate hash of evidence."""
        evidence_json = json.dumps(evidence, sort_keys=True)
        return hashlib.sha256(evidence_json.encode()).hexdigest()
    
    def _calculate_entry_hash(self, entry: AuditEntry) -> str:
        """Calculate hash of an audit entry."""
        entry_data = {
            'entry_id': entry.entry_id,
            'timestamp': entry.timestamp.isoformat(),
            'action_id': entry.action_id,
            'agent_id': entry.agent_id,
            'decision': entry.decision,
            'reason': entry.reason,
            'evidence_hash': entry.evidence_hash,
            'previous_hash': entry.previous_hash
        }
        entry_json = json.dumps(entry_data, sort_keys=True)
        return hashlib.sha256(entry_json.encode()).hexdigest()
    
    def _generate_entry_id(
        self,
        action_id: str,
        agent_id: str,
        timestamp: datetime
    ) -> str:
        """Generate unique entry ID."""
        seed = f"{action_id}:{agent_id}:{timestamp.isoformat()}:{time.time_ns()}"
        return hashlib.sha256(seed.encode()).hexdigest()[:32]
    
    def _update_confirmation_metrics(self, confirmation_time: float):
        """Update confirmation time metrics."""
        old_avg = self.metrics['average_confirmation_time']
        n = self.metrics['blockchain_entries']
        
        if n == 1:
            self.metrics['average_confirmation_time'] = confirmation_time
        else:
            self.metrics['average_confirmation_time'] = (
                (old_avg * (n - 1) + confirmation_time) / n
            )
