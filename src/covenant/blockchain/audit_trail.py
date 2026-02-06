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

logger = logging.getLogger(__name__)

@dataclass
class AuditEntry:
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
    leaf_hash: str
    path: List[Tuple[str, bool]]  # (hash, is_left)
    root_hash: str
    tree_depth: int

class BlockchainAuditTrail:
    def __init__(self, blockchain_config: Optional[Dict[str, Any]] = None, use_ipfs: bool = True):
        self.config = blockchain_config or {}
        self.use_ipfs = use_ipfs

        self.web3 = None
        self.contract = None
        self.account = None
        self.ipfs_client = None

        self.audit_chain: List[AuditEntry] = []
        self.merkle_tree: Dict[str, Dict[str, Any]] = {}
        self.root_hash = ""

        self.metrics = {
            'total_entries': 0,
            'blockchain_entries': 0,
            'ipfs_uploads': 0,
            'verifications': 0,
            'average_confirmation_time': 0.0
        }

        self._initialize_connections()
        logger.info("BlockchainAuditTrail initialized")

    def _initialize_connections(self):
        try:
            if 'provider_url' in self.config:
                self.web3 = Web3(Web3.HTTPProvider(self.config['provider_url']))
                if self.web3.is_connected():
                    logger.info(f"Connected to blockchain at {self.config['provider_url']}")
                    if 'private_key' in self.config:
                        self.account = Account.from_key(self.config['private_key'])
                        logger.info(f"Loaded account: {self.account.address}")
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
        start_time = time.time()
        try:
            evidence_hash = self._hash_evidence(evidence)
            ipfs_cid = None
            if self.ipfs_client and evidence:
                ipfs_cid = await self._store_on_ipfs(evidence)
                self.metrics['ipfs_uploads'] += 1

            previous_hash = self.root_hash if self.audit_chain else "0"*64
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
                metadata={'ipfs_cid': ipfs_cid, **(metadata or {})}
            )

            self.audit_chain.append(entry)
            await self._update_merkle_tree(entry)

            if self.contract and self.account:
                tx_hash = await self._record_on_blockchain(entry)
                if tx_hash:
                    entry.transaction_hash = tx_hash
                    block_number = await self._wait_for_confirmation(tx_hash)
                    if block_number:
                        entry.block_number = block_number
                        self.metrics['blockchain_entries'] += 1
                    self._update_confirmation_metrics(time.time() - start_time)

            self.metrics['total_entries'] += 1
            logger.info(f"Logged decision for action {action_id}: {decision}")
            return entry
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
            raise

    async def _store_on_ipfs(self, evidence: Dict[str, Any]) -> str:
        if not self.ipfs_client:
            raise RuntimeError("IPFS client not available")
        evidence_json = json.dumps(evidence, indent=2)
        result = self.ipfs_client.add_bytes(evidence_json.encode())
        cid = result['Hash']
        logger.debug(f"Stored evidence on IPFS with CID: {cid}")
        return cid

    async def _record_on_blockchain(self, entry: AuditEntry) -> str | None:
        if not self.contract or not self.account:
            return None
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_blockchain_call, entry)

    def _sync_blockchain_call(self, entry: AuditEntry) -> str | None:
        try:
            nonce = self.web3.eth.get_transaction_count(self.account.address)
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
                'gas': 200_000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': nonce,
                'from': self.account.address
            })
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            logger.info(f"Blockchain transaction submitted: {tx_hash.hex()} for entry {entry.entry_id}")
            return tx_hash.hex()
        except Exception as e:
            logger.error(f"Synchronous blockchain call failed: {e}")
            return None

    async def _wait_for_confirmation(self, tx_hash: str, timeout: int = 60) -> Optional[int]:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                receipt = self.web3.eth.get_transaction_receipt(tx_hash)
                if receipt:
                    return receipt.blockNumber
            except: pass
            await asyncio.sleep(1)
        logger.warning(f"Transaction {tx_hash} not confirmed within {timeout} seconds")
        return None

    async def _update_merkle_tree(self, entry: AuditEntry):
        leaf_hash = self._calculate_entry_hash(entry)
        if entry.entry_id not in self.merkle_tree:
            self.merkle_tree[entry.entry_id] = {'hash': leaf_hash, 'index': len(self.merkle_tree)}
        self.root_hash = self._calculate_merkle_root()
        await self._update_merkle_proofs()

    async def _update_merkle_proofs(self):
        leaves = sorted(self.merkle_tree.items(), key=lambda x: x[1]['index'])
        leaf_hashes = [data['hash'] for _, data in leaves]
        for i, (entry_id, data) in enumerate(leaves):
            proof_path = self._calculate_merkle_proof(leaf_hashes, i)
            data['proof'] = MerkleProof(
                leaf_hash=data['hash'],
                path=proof_path,
                root_hash=self.root_hash,
                tree_depth=len(proof_path)
            )

    def _calculate_merkle_root(self) -> str:
        leaves = [data['hash'] for _, data in sorted(self.merkle_tree.items(), key=lambda x: x[1]['index'])]
        if not leaves: return "0"*64
        while len(leaves) > 1:
            if len(leaves) % 2: leaves.append(leaves[-1])
            leaves = [hashlib.sha256((leaves[i]+leaves[i+1]).encode()).hexdigest() for i in range(0, len(leaves), 2)]
        return leaves[0]

    def _calculate_merkle_proof(self, leaves: List[str], index: int) -> List[Tuple[str, bool]]:
        proof = []
        current_index = index
        while len(leaves) > 1:
            if len(leaves) % 2: leaves.append(leaves[-1])
            sibling_index = current_index - 1 if current_index % 2 else current_index + 1
            proof.append((leaves[sibling_index], current_index % 2 == 1))
            current_index //= 2
            leaves = [hashlib.sha256((leaves[i]+leaves[i+1]).encode()).hexdigest() for i in range(0, len(leaves), 2)]
        return proof

    def _hash_evidence(self, evidence: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()

    def _calculate_entry_hash(self, entry: AuditEntry) -> str:
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
        return hashlib.sha256(json.dumps(entry_data, sort_keys=True).encode()).hexdigest()

    def _generate_entry_id(self, action_id: str, agent_id: str, timestamp: datetime) -> str:
        seed = f"{action_id}:{agent_id}:{timestamp.isoformat()}:{time.time_ns()}"
        return hashlib.sha256(seed.encode()).hexdigest()[:32]

    def _update_confirmation_metrics(self, confirmation_time: float):
        n = self.metrics['blockchain_entries']
        old_avg = self.metrics['average_confirmation_time']
        self.metrics['average_confirmation_time'] = ((old_avg*(n-1))+confirmation_time)/n if n>1 else confirmation_time
