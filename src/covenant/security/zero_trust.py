"""
Zero-trust security model for Covenant.AI with continuous verification.
"""

import logging
import hashlib
import hmac
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
import os

from covenant.security.encryption import EncryptionManager

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for zero-trust model."""
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"

class ThreatLevel(Enum):
    """Threat levels for security monitoring."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityPolicy:
    """Security policy for zero-trust model."""
    policy_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    security_level: SecurityLevel
    allowed_principals: List[str]
    allowed_actions: List[str]
    constraints: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Principal:
    """Security principal (user, service, agent)."""
    principal_id: str
    principal_type: str  # user, service, agent
    security_clearance: SecurityLevel
    attributes: Dict[str, Any]
    credentials_hash: str
    last_authentication: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityEvent:
    """Security event for monitoring."""
    event_id: str
    timestamp: datetime
    principal_id: str
    action: str
    resource: str
    outcome: str  # allowed, denied, suspicious
    threat_level: ThreatLevel
    details: Dict[str, Any]
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ZeroTrustSecurity:
    """
    Zero-trust security implementation with continuous verification.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_continuous_verification: bool = True
    ):
        """
        Initialize zero-trust security.
        
        Args:
            config: Security configuration
            enable_continuous_verification: Whether to enable continuous verification
        """
        self.config = config or {}
        self.enable_continuous_verification = enable_continuous_verification
        
        # Security state
        self.principals: Dict[str, Principal] = {}
        self.policies: Dict[str, SecurityPolicy] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.security_events: List[SecurityEvent] = []
        
        # Encryption manager
        self.encryption = EncryptionManager()
        
        # Anomaly detection
        self.behavior_profiles: Dict[str, Dict[str, Any]] = {}
        self.threat_intelligence: List[Dict[str, Any]] = []
        
        # Continuous verification
        self.verification_tasks: Dict[str, asyncio.Task] = {}
        
        # Security metrics
        self.metrics = {
            'total_requests': 0,
            'allowed_requests': 0,
            'denied_requests': 0,
            'suspicious_events': 0,
            'authentication_failures': 0,
            'average_risk_score': 0.0,
            'threat_detections': 0
        }
        
        # Load default policies
        self._load_default_policies()
        
        logger.info("ZeroTrustSecurity initialized")
    
    def _load_default_policies(self):
        """Load default security policies."""
        # Default policy: Constitutional AI Operations
        constitutional_policy = SecurityPolicy(
            policy_id="constitutional_ai_ops",
            name="Constitutional AI Operations",
            description="Policy for constitutional AI decision-making",
            rules=[
                {
                    "rule_id": "constitutional_verification",
                    "description": "Require constitutional verification for all AI actions",
                    "condition": "action_type in ['ai_decision', 'ai_action', 'model_inference']",
                    "effect": "allow"
                },
                {
                    "rule_id": "audit_logging",
                    "description": "Require audit logging for all security-relevant actions",
                    "condition": "security_level >= SecurityLevel.CONFIDENTIAL",
                    "effect": "allow"
                }
            ],
            security_level=SecurityLevel.SECRET,
            allowed_principals=["*"],  # All authenticated principals
            allowed_actions=["ai_decision", "model_inference", "data_processing"],
            constraints={
                "max_concurrent_actions": 10,
                "require_multi_factor": True,
                "session_timeout": 3600
            }
        )
        
        self.policies[constitutional_policy.policy_id] = constitutional_policy
        
        # Default policy: System Administration
        admin_policy = SecurityPolicy(
            policy_id="system_admin",
            name="System Administration",
            description="Policy for system administration tasks",
            rules=[
                {
                    "rule_id": "admin_access",
                    "description": "Allow administrative access",
                    "condition": "principal.attributes.is_admin == True",
                    "effect": "allow"
                }
            ],
            security_level=SecurityLevel.TOP_SECRET,
            allowed_principals=["admin_users"],
            allowed_actions=["system_administration", "configuration", "user_management"],
            constraints={
                "require_multi_factor": True,
                "ip_whitelist": ["10.0.0.0/8", "192.168.0.0/16"],
                "time_restrictions": ["09:00-17:00"]
            }
        )
        
        self.policies[admin_policy.policy_id] = admin_policy
        
        logger.info(f"Loaded {len(self.policies)} default security policies")
    
    async def authenticate_principal(
        self,
        principal_id: str,
        credentials: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Authenticate a principal with multi-factor verification.
        
        Args:
            principal_id: Principal identifier
            credentials: Authentication credentials
            context: Authentication context
            
        Returns:
            Tuple of (success, session_token, auth_result)
        """
        self.metrics['total_requests'] += 1
        
        try:
            # Check if principal exists
            if principal_id not in self.principals:
                logger.warning(f"Unknown principal: {principal_id}")
                self.metrics['authentication_failures'] += 1
                
                await self.log_security_event(
                    principal_id=principal_id,
                    action="authentication",
                    resource="system",
                    outcome="denied",
                    threat_level=ThreatLevel.MEDIUM,
                    details={"reason": "unknown_principal"}
                )
                
                return False, None, {"error": "Unknown principal"}
            
            principal = self.principals[principal_id]
            
            if not principal.is_active:
                logger.warning(f"Inactive principal: {principal_id}")
                self.metrics['authentication_failures'] += 1
                
                await self.log_security_event(
                    principal_id=principal_id,
                    action="authentication",
                    resource="system",
                    outcome="denied",
                    threat_level=ThreatLevel.MEDIUM,
                    details={"reason": "inactive_principal"}
                )
                
                return False, None, {"error": "Principal is inactive"}
            
            # Verify credentials
            auth_result = await self._verify_credentials(principal, credentials, context)
            
            if not auth_result['success']:
                self.metrics['authentication_failures'] += 1
                
                await self.log_security_event(
                    principal_id=principal_id,
                    action="authentication",
                    resource="system",
                    outcome="denied",
                    threat_level=ThreatLevel.HIGH if auth_result.get('suspicious') else ThreatLevel.MEDIUM,
                    details=auth_result
                )
                
                return False, None, auth_result
            
            # Multi-factor verification if required
            if auth_result.get('require_mfa', False):
                mfa_result = await self._verify_multi_factor(principal, credentials, context)
                
                if not mfa_result['success']:
                    self.metrics['authentication_failures'] += 1
                    
                    await self.log_security_event(
                        principal_id=principal_id,
                        action="mfa_verification",
                        resource="system",
                        outcome="denied",
                        threat_level=ThreatLevel.HIGH,
                        details=mfa_result
                    )
                    
                    return False, None, mfa_result
            
            # Generate session token
            session_token = await self._create_session(principal, context)
            
            # Update principal
            principal.last_authentication = datetime.utcnow()
            
            # Start continuous verification if enabled
            if self.enable_continuous_verification:
                await self._start_continuous_verification(principal_id, session_token)
            
            self.metrics['allowed_requests'] += 1
            
            await self.log_security_event(
                principal_id=principal_id,
                action="authentication",
                resource="system",
                outcome="allowed",
                threat_level=ThreatLevel.LOW,
                details={"method": auth_result.get('method', 'unknown')}
            )
            
            return True, session_token, auth_result
            
        except Exception as e:
            logger.error(f"Authentication error for {principal_id}: {e}")
            
            await self.log_security_event(
                principal_id=principal_id,
                action="authentication",
                resource="system",
                outcome="denied",
                threat_level=ThreatLevel.CRITICAL,
                details={"error": str(e)}
            )
            
            return False, None, {"error": str(e)}
    
    async def authorize_action(
        self,
        principal_id: str,
        action: str,
        resource: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Authorize an action for a principal.
        
        Args:
            principal_id: Principal identifier
            action: Action to authorize
            resource: Resource being accessed
            context: Authorization context
            
        Returns:
            Tuple of (allowed, policy_id, authz_result)
        """
        self.metrics['total_requests'] += 1
        
        try:
            # Check if principal is authenticated
            if principal_id not in self.principals:
                logger.warning(f"Unknown principal: {principal_id}")
                self.metrics['denied_requests'] += 1
                
                await self.log_security_event(
                    principal_id=principal_id,
                    action=action,
                    resource=resource,
                    outcome="denied",
                    threat_level=ThreatLevel.MEDIUM,
                    details={"reason": "unknown_principal"}
                )
                
                return False, None, {"error": "Unknown principal"}
            
            principal = self.principals[principal_id]
            
            # Check session validity
            session_valid = await self._verify_session(principal_id, context)
            if not session_valid:
                self.metrics['denied_requests'] += 1
                
                await self.log_security_event(
                    principal_id=principal_id,
                    action=action,
                    resource=resource,
                    outcome="denied",
                    threat_level=ThreatLevel.HIGH,
                    details={"reason": "invalid_session"}
                )
                
                return False, None, {"error": "Invalid or expired session"}
            
            # Find applicable policies
            applicable_policies = await self._find_applicable_policies(
                principal, action, resource, context
            )
            
            if not applicable_policies:
                logger.warning(f"No applicable policies for {action} on {resource}")
                self.metrics['denied_requests'] += 1
                
                await self.log_security_event(
                    principal_id=principal_id,
                    action=action,
                    resource=resource,
                    outcome="denied",
                    threat_level=ThreatLevel.MEDIUM,
                    details={"reason": "no_applicable_policies"}
                )
                
                return False, None, {"error": "No applicable policies"}
            
            # Evaluate policies
            authz_result = await self._evaluate_policies(
                principal, action, resource, applicable_policies, context
            )
            
            if authz_result['allowed']:
                self.metrics['allowed_requests'] += 1
                
                # Update behavior profile
                await self._update_behavior_profile(principal_id, action, resource, True)
                
                await self.log_security_event(
                    principal_id=principal_id,
                    action=action,
                    resource=resource,
                    outcome="allowed",
                    threat_level=ThreatLevel.LOW,
                    details=authz_result
                )
                
                return True, authz_result.get('policy_id'), authz_result
            else:
                self.metrics['denied_requests'] += 1
                
                # Check if suspicious
                risk_score = authz_result.get('risk_score', 0)
                if risk_score > 0.7:
                    self.metrics['suspicious_events'] += 1
                    
                    await self.log_security_event(
                        principal_id=principal_id,
                        action=action,
                        resource=resource,
                        outcome="denied",
                        threat_level=ThreatLevel.HIGH,
                        details=authz_result
                    )
                else:
                    await self.log_security_event(
                        principal_id=principal_id,
                        action=action,
                        resource=resource,
                        outcome="denied",
                        threat_level=ThreatLevel.MEDIUM,
                        details=authz_result
                    )
                
                return False, None, authz_result
            
        except Exception as e:
            logger.error(f"Authorization error for {principal_id}: {e}")
            
            await self.log_security_event(
                principal_id=principal_id,
                action=action,
                resource=resource,
                outcome="denied",
                threat_level=ThreatLevel.CRITICAL,
                details={"error": str(e)}
            )
            
            return False, None, {"error": str(e)}
    
    async def register_principal(
        self,
        principal_id: str,
        principal_type: str,
        attributes: Dict[str, Any],
        credentials: Dict[str, Any],
        security_clearance: SecurityLevel = SecurityLevel.CONFIDENTIAL
    ) -> Tuple[bool, str]:
        """
        Register a new principal.
        
        Args:
            principal_id: Principal identifier
            principal_type: Type of principal
            attributes: Principal attributes
            credentials: Initial credentials
            security_clearance: Security clearance level
            
        Returns:
            Tuple of (success, message)
        """
        if principal_id in self.principals:
            return False, f"Principal {principal_id} already exists"
        
        try:
            # Hash credentials
            credentials_hash = self._hash_credentials(credentials)
            
            # Create principal
            principal = Principal(
                principal_id=principal_id,
                principal_type=principal_type,
                security_clearance=security_clearance,
                attributes=attributes,
                credentials_hash=credentials_hash,
                last_authentication=datetime.utcnow()
            )
            
            self.principals[principal_id] = principal
            
            # Initialize behavior profile
            self.behavior_profiles[principal_id] = {
                'created_at': datetime.utcnow(),
                'action_counts': {},
                'resource_access': {},
                'risk_history': [],
                'anomaly_score': 0.0
            }
            
            logger.info(f"Registered principal {principal_id} with clearance {security_clearance.value}")
            
            await self.log_security_event(
                principal_id=principal_id,
                action="registration",
                resource="system",
                outcome="allowed",
                threat_level=ThreatLevel.LOW,
                details={"type": principal_type, "clearance": security_clearance.value}
            )
            
            return True, f"Principal {principal_id} registered successfully"
            
        except Exception as e:
            logger.error(f"Failed to register principal {principal_id}: {e}")
            return False, str(e)
    
    async def add_security_policy(self, policy: SecurityPolicy) -> bool:
        """Add a new security policy."""
        if policy.policy_id in self.policies:
            logger.warning(f"Policy {policy.policy_id} already exists")
            return False
        
        self.policies[policy.policy_id] = policy
        logger.info(f"Added security policy: {policy.name}")
        return True
    
    async def remove_security_policy(self, policy_id: str) -> bool:
        """Remove a security policy."""
        if policy_id in self.policies:
            del self.policies[policy_id]
            logger.info(f"Removed security policy: {policy_id}")
            return True
        else:
            logger.warning(f"Policy {policy_id} not found")
            return False
    
    async def log_security_event(
        self,
        principal_id: str,
        action: str,
        resource: str,
        outcome: str,
        threat_level: ThreatLevel,
        details: Dict[str, Any]
    ):
        """Log a security event."""
        event = SecurityEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.utcnow(),
            principal_id=principal_id,
            action=action,
            resource=resource,
            outcome=outcome,
            threat_level=threat_level,
            details=details,
            risk_score=details.get('risk_score', 0.0)
        )
        
        self.security_events.append(event)
        
        # Keep only last 10,000 events
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-10000:]
        
        # Check for threat patterns
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            await self._analyze_threat_patterns(event)
        
        logger.debug(f"Logged security event: {event.event_id} - {action} on {resource} - {outcome}")
    
    async def get_security_events(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[SecurityEvent]:
        """Get security events with filters."""
        filtered_events = self.security_events.copy()
        
        if filters:
            for key, value in filters.items():
                if key == 'principal_id':
                    filtered_events = [e for e in filtered_events if e.principal_id == value]
                elif key == 'action':
                    filtered_events = [e for e in filtered_events if e.action == value]
                elif key == 'outcome':
                    filtered_events = [e for e in filtered_events if e.outcome == value]
                elif key == 'threat_level':
                    filtered_events = [e for e in filtered_events if e.threat_level == value]
                elif key == 'start_time':
                    filtered_events = [e for e in filtered_events if e.timestamp >= value]
                elif key == 'end_time':
                    filtered_events = [e for e in filtered_events if e.timestamp <= value]
                elif key == 'min_risk_score':
                    filtered_events = [e for e in filtered_events if e.risk_score >= value]
        
        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        
        return filtered_events[start_idx:end_idx]
    
    async def get_security_report(self) -> Dict[str, Any]:
        """Get security report with statistics."""
        total_events = len(self.security_events)
        
        if total_events == 0:
            return {
                'total_events': 0,
                'event_distribution': {},
                'threat_levels': {},
                'top_principals': [],
                'recent_alerts': []
            }
        
        # Event distribution by outcome
        outcomes = {}
        threat_levels = {}
        principal_counts = {}
        
        for event in self.security_events:
            # Outcomes
            outcomes[event.outcome] = outcomes.get(event.outcome, 0) + 1
            
            # Threat levels
            threat_levels[event.threat_level.value] = threat_levels.get(event.threat_level.value, 0) + 1
            
            # Principals
            principal_counts[event.principal_id] = principal_counts.get(event.principal_id, 0) + 1
        
        # Top principals by event count
        top_principals = sorted(
            principal_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Recent high/critical threats
        recent_alerts = [
            {
                'event_id': e.event_id,
                'timestamp': e.timestamp.isoformat(),
                'principal_id': e.principal_id,
                'action': e.action,
                'threat_level': e.threat_level.value,
                'risk_score': e.risk_score
            }
            for e in self.security_events[-100:]  # Last 100 events
            if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        ][-10:]  # Last 10 alerts
        
        return {
            'total_events': total_events,
            'event_distribution': outcomes,
            'threat_levels': threat_levels,
            'top_principals': top_principals,
            'recent_alerts': recent_alerts,
            'metrics': self.metrics.copy()
        }
    
    async def _verify_credentials(
        self,
        principal: Principal,
        credentials: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify principal credentials."""
        result = {
            'success': False,
            'method': 'unknown',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Check password
            if 'password' in credentials:
                provided_hash = self._hash_credentials({'password': credentials['password']})
                
                if hmac.compare_digest(provided_hash, principal.credentials_hash):
                    result['success'] = True
                    result['method'] = 'password'
                    
                    # Check if MFA is required
                    if principal.security_clearance.value >= SecurityLevel.SECRET.value:
                        result['require_mfa'] = True
                else:
                    result['error'] = 'Invalid password'
            
            # Check API key
            elif 'api_key' in credentials:
                # In production, use proper API key validation
                if credentials.get('api_key') == 'valid_api_key':  # Placeholder
                    result['success'] = True
                    result['method'] = 'api_key'
                else:
                    result['error'] = 'Invalid API key'
            
            # Check JWT token
            elif 'jwt_token' in credentials:
                try:
                    # Verify JWT
                    decoded = jwt.decode(
                        credentials['jwt_token'],
                        self.config.get('jwt_secret', 'secret'),
                        algorithms=['HS256']
                    )
                    
                    if decoded.get('sub') == principal.principal_id:
                        result['success'] = True
                        result['method'] = 'jwt'
                    else:
                        result['error'] = 'Invalid JWT subject'
                except jwt.InvalidTokenError as e:
                    result['error'] = f'Invalid JWT: {str(e)}'
            
            else:
                result['error'] = 'No valid credentials provided'
            
            # Check for suspicious patterns
            if not result['success']:
                result['suspicious'] = await self._check_suspicious_authentication(
                    principal, credentials, context
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Credential verification error: {e}")
            result['error'] = str(e)
            return result
    
    async def _verify_multi_factor(
        self,
        principal: Principal,
        credentials: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify multi-factor authentication."""
        result = {
            'success': False,
            'method': 'unknown',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Check for TOTP
            if 'totp_code' in credentials:
                # In production, use proper TOTP validation
                if credentials.get('totp_code') == '123456':  # Placeholder
                    result['success'] = True
                    result['method'] = 'totp'
                else:
                    result['error'] = 'Invalid TOTP code'
            
            # Check for hardware token
            elif 'hardware_token' in credentials:
                # In production, use proper hardware token validation
                if credentials.get('hardware_token') == 'valid_token':  # Placeholder
                    result['success'] = True
                    result['method'] = 'hardware_token'
                else:
                    result['error'] = 'Invalid hardware token'
            
            # Check for biometric
            elif 'biometric_data' in credentials:
                # In production, use proper biometric validation
                if credentials.get('biometric_data') == 'valid_biometric':  # Placeholder
                    result['success'] = True
                    result['method'] = 'biometric'
                else:
                    result['error'] = 'Invalid biometric data'
            
            else:
                result['error'] = 'No MFA credentials provided'
            
            return result
            
        except Exception as e:
            logger.error(f"MFA verification error: {e}")
            result['error'] = str(e)
            return result
    
    async def _create_session(
        self,
        principal: Principal,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Create a new session for the principal."""
        session_id = self._generate_session_id(principal.principal_id)
        
        session_data = {
            'session_id': session_id,
            'principal_id': principal.principal_id,
            'created_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'security_clearance': principal.security_clearance.value,
            'ip_address': context.get('ip_address') if context else None,
            'user_agent': context.get('user_agent') if context else None,
            'expires_at': datetime.utcnow() + timedelta(hours=1)
        }
        
        # Encrypt session data
        encrypted_session = self.encryption.encrypt(json.dumps(session_data, default=str))
        
        # Store session
        self.sessions[session_id] = {
            'data': session_data,
            'encrypted': encrypted_session
        }
        
        return session_id
    
    async def _verify_session(
        self,
        principal_id: str,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Verify session validity."""
        if not context or 'session_token' not in context:
            return False
        
        session_token = context['session_token']
        
        if session_token not in self.sessions:
            return False
        
        session = self.sessions[session_token]
        session_data = session['data']
        
        # Check expiration
        if session_data['expires_at'] < datetime.utcnow():
            # Remove expired session
            del self.sessions[session_token]
            return False
        
        # Check principal match
        if session_data['principal_id'] != principal_id:
            return False
        
        # Check IP address if provided in context
        if 'ip_address' in context and session_data['ip_address']:
            if context['ip_address'] != session_data['ip_address']:
                # Log suspicious IP change
                await self.log_security_event(
                    principal_id=principal_id,
                    action="session_verification",
                    resource="session",
                    outcome="suspicious",
                    threat_level=ThreatLevel.MEDIUM,
                    details={
                        'reason': 'ip_address_change',
                        'old_ip': session_data['ip_address'],
                        'new_ip': context['ip_address']
                    }
                )
                # Allow with warning for now
                # return False
        
        # Update last activity
        session_data['last_activity'] = datetime.utcnow()
        
        return True
    
    async def _find_applicable_policies(
        self,
        principal: Principal,
        action: str,
        resource: str,
        context: Optional[Dict[str, Any]]
    ) -> List[SecurityPolicy]:
        """Find policies applicable to the request."""
        applicable_policies = []
        
        for policy in self.policies.values():
            # Check if principal is allowed
            if (policy.allowed_principals != ["*"] and 
                principal.principal_id not in policy.allowed_principals):
                continue
            
            # Check if action is allowed
            if action not in policy.allowed_actions:
                continue
            
            # Check security clearance
            if principal.security_clearance.value < policy.security_level.value:
                continue
            
            applicable_policies.append(policy)
        
        return applicable_policies
    
    async def _evaluate_policies(
        self,
        principal: Principal,
        action: str,
        resource: str,
        policies: List[SecurityPolicy],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate policies and make authorization decision."""
        result = {
            'allowed': False,
            'policies_evaluated': [],
            'risk_score': 0.0,
            'constraints': {}
        }
        
        try:
            # Check behavior anomalies
            anomaly_score = await self._check_behavior_anomalies(
                principal.principal_id, action, resource, context
            )
            
            if anomaly_score > 0.8:
                result['risk_score'] = 1.0
                result['denied_reason'] = 'suspicious_behavior'
                return result
            
            # Evaluate each policy
            for policy in policies:
                policy_result = await self._evaluate_policy(
                    principal, action, resource, policy, context
                )
                
                result['policies_evaluated'].append({
                    'policy_id': policy.policy_id,
                    'allowed': policy_result['allowed'],
                    'rules_matched': policy_result['rules_matched']
                })
                
                if policy_result['allowed']:
                    result['allowed'] = True
                    result['policy_id'] = policy.policy_id
                    result['constraints'].update(policy.constraints)
                    
                    # Apply risk score from policy evaluation
                    if 'risk_score' in policy_result:
                        result['risk_score'] = max(
                            result['risk_score'],
                            policy_result['risk_score']
                        )
                    
                    break  # First allowed policy grants access
            
            # If no policy allows, access is denied
            if not result['allowed']:
                result['denied_reason'] = 'no_policy_allows'
                result['risk_score'] = 0.5  # Default risk for denied access
            
            return result
            
        except Exception as e:
            logger.error(f"Policy evaluation error: {e}")
            result['denied_reason'] = 'evaluation_error'
            result['risk_score'] = 1.0
            return result
    
    async def _evaluate_policy(
        self,
        principal: Principal,
        action: str,
        resource: str,
        policy: SecurityPolicy,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a single policy."""
        result = {
            'allowed': False,
            'rules_matched': [],
            'risk_score': 0.0
        }
        
        try:
            # Evaluate each rule
            for rule in policy.rules:
                rule_result = await self._evaluate_rule(
                    principal, action, resource, rule, context
                )
                
                if rule_result['matched']:
                    result['rules_matched'].append(rule['rule_id'])
                    
                    if rule['effect'] == 'allow':
                        result['allowed'] = True
                    
                    # Update risk score
                    if 'risk_score' in rule_result:
                        result['risk_score'] = max(
                            result['risk_score'],
                            rule_result['risk_score']
                        )
            
            return result
            
        except Exception as e:
            logger.error(f"Rule evaluation error: {e}")
            result['risk_score'] = 1.0
            return result
    
    async def _evaluate_rule(
        self,
        principal: Principal,
        action: str,
        resource: str,
        rule: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a single rule."""
        result = {
            'matched': False,
            'risk_score': 0.0
        }
        
        try:
            # Parse condition
            condition = rule.get('condition', '')
            
            if not condition:
                # Empty condition always matches
                result['matched'] = True
                return result
            
            # Create evaluation context
            eval_context = {
                'principal': principal,
                'action': action,
                'resource': resource,
                'context': context or {},
                'SecurityLevel': SecurityLevel
            }
            
            # Evaluate condition (simplified)
            # In production, use a proper safe evaluator
            if 'action_type' in condition:
                # Simplified condition evaluation
                result['matched'] = True
            elif 'security_level' in condition:
                # Check security level
                required_level = condition.split('>=')[1].strip()
                required_level = SecurityLevel[required_level.split('.')[1]]
                result['matched'] = (
                    principal.security_clearance.value >= required_level.value
                )
            else:
                # Default: condition not recognized, don't match
                result['matched'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            result['risk_score'] = 1.0
            return result
    
    async def _check_behavior_anomalies(
        self,
        principal_id: str,
        action: str,
        resource: str,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Check for behavioral anomalies."""
        if principal_id not in self.behavior_profiles:
            return 0.0
        
        profile = self.behavior_profiles[principal_id]
        
        # Check action frequency
        action_count = profile['action_counts'].get(action, 0)
        if action_count > 100:  # Threshold
            return 0.7
        
        # Check time of day anomalies
        if context and 'timestamp' in context:
            hour = datetime.fromisoformat(context['timestamp']).hour
            if hour < 6 or hour > 22:  # Unusual hours
                return 0.6
        
        # Check resource access pattern
        if resource not in profile['resource_access']:
            # New resource access
            return 0.4
        
        return 0.0
    
    async def _update_behavior_profile(
        self,
        principal_id: str,
        action: str,
        resource: str,
        allowed: bool
    ):
        """Update behavior profile for a principal."""
        if principal_id not in self.behavior_profiles:
            return
        
        profile = self.behavior_profiles[principal_id]
        
        # Update action count
        profile['action_counts'][action] = profile['action_counts'].get(action, 0) + 1
        
        # Update resource access
        if resource not in profile['resource_access']:
            profile['resource_access'][resource] = []
        
        profile['resource_access'][resource].append({
            'timestamp': datetime.utcnow(),
            'allowed': allowed
        })
        
        # Calculate anomaly score
        profile['anomaly_score'] = await self._calculate_anomaly_score(profile)
    
    async def _calculate_anomaly_score(self, profile: Dict[str, Any]) -> float:
        """Calculate anomaly score for behavior profile."""
        score = 0.0
        
        # High action frequency penalty
        total_actions = sum(profile['action_counts'].values())
        if total_actions > 1000:
            score += 0.3
        
        # New resource access penalty
        new_resources = len([
            r for r, accesses in profile['resource_access'].items()
            if len(accesses) < 3
        ])
        if new_resources > 10:
            score += 0.2
        
        # Unusual time pattern (simplified)
        # In production, use proper time series analysis
        
        return min(score, 1.0)
    
    async def _check_suspicious_authentication(
        self,
        principal: Principal,
        credentials: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Check for suspicious authentication patterns."""
        suspicious = False
        
        # Multiple failed attempts
        failed_attempts = len([
            e for e in self.security_events[-100:]
            if e.principal_id == principal.principal_id
            and e.action == 'authentication'
            and e.outcome == 'denied'
        ])
        
        if failed_attempts > 5:
            suspicious = True
        
        # Unusual location
        if context and 'ip_address' in context:
            # In production, check against known locations
            pass
        
        return suspicious
    
    async def _analyze_threat_patterns(self, event: SecurityEvent):
        """Analyze security events for threat patterns."""
        # Look for patterns in recent events
        recent_events = self.security_events[-100:]
        
        # Check for brute force attacks
        failed_auths = [
            e for e in recent_events
            if e.action == 'authentication' and e.outcome == 'denied'
        ]
        
        if len(failed_auths) > 10:
            logger.warning(f"Possible brute force attack detected: {len(failed_auths)} failed authentications")
            self.metrics['threat_detections'] += 1
        
        # Check for privilege escalation attempts
        privilege_events = [
            e for e in recent_events
            if e.action in ['policy_modification', 'user_escalation']
            and e.outcome == 'denied'
        ]
        
        if len(privilege_events) > 5:
            logger.warning(f"Possible privilege escalation attempts: {len(privilege_events)} events")
            self.metrics['threat_detections'] += 1
    
    async def _start_continuous_verification(self, principal_id: str, session_token: str):
        """Start continuous verification for a session."""
        if principal_id in self.verification_tasks:
            # Cancel existing task
            self.verification_tasks[principal_id].cancel()
        
        # Start new verification task
        task = asyncio.create_task(self._continuous_verification_loop(principal_id, session_token))
        self.verification_tasks[principal_id] = task
    
    async def _continuous_verification_loop(self, principal_id: str, session_token: str):
        """Continuous verification loop for a session."""
        try:
            while True:
                await asyncio.sleep(30)  # Verify every 30 seconds
                
                # Check session validity
                if session_token not in self.sessions:
                    break
                
                session = self.sessions[session_token]
                session_data = session['data']
                
                # Check for anomalies
                anomaly_score = await self._check_session_anomalies(session_data)
                
                if anomaly_score > 0.8:
                    # Terminate suspicious session
                    del self.sessions[session_token]
                    
                    await self.log_security_event(
                        principal_id=principal_id,
                        action="session_termination",
                        resource="session",
                        outcome="suspicious",
                        threat_level=ThreatLevel.HIGH,
                        details={'reason': 'continuous_verification_failed'}
                    )
                    
                    break
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Continuous verification error: {e}")
    
    async def _check_session_anomalies(self, session_data: Dict[str, Any]) -> float:
        """Check for session anomalies."""
        score = 0.0
        
        # Check session duration
        session_duration = (datetime.utcnow() - session_data['created_at']).total_seconds()
        if session_duration > 3600 * 8:  # 8 hours
            score += 0.3
        
        # Check inactivity
        inactivity = (datetime.utcnow() - session_data['last_activity']).total_seconds()
        if inactivity > 1800:  # 30 minutes
            score += 0.4
        
        return score
    
    def _hash_credentials(self, credentials: Dict[str, Any]) -> str:
        """Hash credentials for storage."""
        credentials_str = json.dumps(credentials, sort_keys=True)
        salt = self.config.get('salt', 'covenant_salt')
        
        # Use PBKDF2 for password hashing
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
            backend=default_backend()
        )
        
        key = kdf.derive(credentials_str.encode())
        return base64.b64encode(key).decode()
    
    def _generate_session_id(self, principal_id: str) -> str:
        """Generate a unique session ID."""
        seed = f"{principal_id}:{time.time_ns()}:{os.urandom(16).hex()}"
        return hashlib.sha256(seed.encode()).hexdigest()[:32]
    
    def _generate_event_id(self) -> str:
        """Generate a unique event ID."""
        seed = f"event:{time.time_ns()}:{os.urandom(16).hex()}"
        return hashlib.sha256(seed.encode()).hexdigest()[:32]
