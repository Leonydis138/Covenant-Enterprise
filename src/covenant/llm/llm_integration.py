"""
LLM integration with constitutional guardrails and advanced prompting.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import hashlib
import time

import openai
import anthropic
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from covenant.llm.guardrails import Guardrails
from covenant.llm.prompt_engine import PromptEngine

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    tokens_used: int
    finish_reason: str
    safety_scores: Dict[str, float]
    constitutional_check: bool
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMRequest:
    """Request to an LLM."""
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False
    constitutional_check: bool = True
    guardrail_level: str = "standard"
    metadata: Dict[str, Any] = field(default_factory=dict)

class LLMIntegration:
    """
    Unified LLM integration with multiple providers and constitutional guardrails.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        providers: Optional[List[str]] = None
    ):
        """
        Initialize LLM integration.
        
        Args:
            config: Configuration dictionary
            providers: List of LLM providers to enable
        """
        self.config = config or {}
        
        # Initialize providers
        self.providers = providers or ["openai", "anthropic", "google", "huggingface"]
        self.available_providers = []
        
        # Initialize components
        self.guardrails = Guardrails()
        self.prompt_engine = PromptEngine()
        
        # Initialize provider clients
        self._init_providers()
        
        # Response cache
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.response_cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        
        # Rate limiting
        self.rate_limits = {}
        self.request_timestamps = {}
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'tokens_used': 0,
            'average_response_time': 0.0,
            'constitutional_blocks': 0,
            'guardrail_triggers': 0
        }
        
        logger.info(f"LLMIntegration initialized with providers: {self.available_providers}")
    
    def _init_providers(self):
        """Initialize LLM provider clients."""
        # OpenAI
        if "openai" in self.providers:
            try:
                openai.api_key = self.config.get('openai_api_key')
                if openai.api_key:
                    self.openai_client = openai.OpenAI()
                    self.available_providers.append("openai")
                    logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Anthropic
        if "anthropic" in self.providers:
            try:
                anthropic_api_key = self.config.get('anthropic_api_key')
                if anthropic_api_key:
                    self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
                    self.available_providers.append("anthropic")
                    logger.info("Anthropic provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic: {e}")
        
        # Google
        if "google" in self.providers:
            try:
                google_api_key = self.config.get('google_api_key')
                if google_api_key:
                    genai.configure(api_key=google_api_key)
                    self.google_client = genai
                    self.available_providers.append("google")
                    logger.info("Google provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Google: {e}")
        
        # Hugging Face
        if "huggingface" in self.providers:
            try:
                hf_token = self.config.get('hf_token')
                self.hf_models = self.config.get('hf_models', [])
                
                if self.hf_models:
                    self.hf_tokenizer = None
                    self.hf_model = None
                    self.available_providers.append("huggingface")
                    logger.info("Hugging Face provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Hugging Face: {e}")
    
    async def generate(
        self,
        request: LLMRequest,
        provider: Optional[str] = None,
        fallback: bool = True
    ) -> LLMResponse:
        """
        Generate a response from an LLM with constitutional guardrails.
        
        Args:
            request: LLM request
            provider: Specific provider to use
            fallback: Whether to fall back to other providers if the first fails
            
        Returns:
            LLM response with safety checks
        """
        self.metrics['total_requests'] += 1
        start_time = time.time()
        
        try:
            # Apply constitutional guardrails to the prompt
            if request.constitutional_check:
                guardrail_result = await self.guardrails.check_prompt(
                    prompt=request.prompt,
                    system_prompt=request.system_prompt,
                    level=request.guardrail_level
                )
                
                if not guardrail_result.allowed:
                    self.metrics['constitutional_blocks'] += 1
                    logger.warning(f"Prompt blocked by constitutional guardrails: {guardrail_result.reason}")
                    
                    return LLMResponse(
                        content="I cannot respond to this request as it violates constitutional constraints.",
                        model="guardrails",
                        tokens_used=0,
                        finish_reason="guardrail_blocked",
                        safety_scores=guardrail_result.safety_scores,
                        constitutional_check=False,
                        reasoning=guardrail_result.reason,
                        metadata={'guardrail_result': guardrail_result}
                    )
            
            # Enhance prompt with constitutional context
            enhanced_prompt = await self.prompt_engine.enhance_prompt(
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                guardrail_level=request.guardrail_level
            )
            
            # Check cache
            cache_key = self._generate_cache_key(enhanced_prompt, request)
            if self.cache_enabled and cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key]
                
                # Check if cache is still valid
                if time.time() - cached_response['timestamp'] < self.cache_ttl:
                    logger.debug(f"Using cached response for prompt hash: {cache_key[:16]}")
                    return cached_response['response']
            
            # Select provider
            selected_provider = provider or self._select_provider(request)
            
            if not selected_provider:
                raise RuntimeError("No LLM providers available")
            
            # Apply rate limiting
            await self._apply_rate_limit(selected_provider)
            
            # Generate response
            response = await self._call_provider(
                provider=selected_provider,
                prompt=enhanced_prompt,
                request=request
            )
            
            # Apply constitutional guardrails to the response
            if request.constitutional_check:
                response_check = await self.guardrails.check_response(
                    prompt=request.prompt,
                    response=response.content,
                    level=request.guardrail_level
                )
                
                if not response_check.allowed:
                    self.metrics['guardrail_triggers'] += 1
                    logger.warning(f"Response filtered by constitutional guardrails: {response_check.reason}")
                    
                    # Replace with safe response
                    safe_response = await self.guardrails.generate_safe_response(
                        original_response=response.content,
                        violation_type=response_check.violation_type
                    )
                    
                    response.content = safe_response
                    response.safety_scores = response_check.safety_scores
                    response.constitutional_check = False
                    response.reasoning = response_check.reason
            
            # Update metrics
            response_time = time.time() - start_time
            self._update_metrics(response_time, response.tokens_used, True)
            
            # Cache response
            if self.cache_enabled:
                self.response_cache[cache_key] = {
                    'response': response,
                    'timestamp': time.time()
                }
            
            self.metrics['successful_requests'] += 1
            return response
            
        except Exception as e:
            # Try fallback if enabled
            if fallback and provider and len(self.available_providers) > 1:
                logger.warning(f"Provider {provider} failed, trying fallback: {e}")
                
                # Remove failed provider from available options
                fallback_providers = [p for p in self.available_providers if p != provider]
                
                for fallback_provider in fallback_providers:
                    try:
                        return await self.generate(request, fallback_provider, fallback=False)
                    except:
                        continue
            
            # All providers failed
            self.metrics['failed_requests'] += 1
            self._update_metrics(time.time() - start_time, 0, False)
            
            logger.error(f"All LLM providers failed: {e}")
            raise
    
    async def _call_provider(
        self,
        provider: str,
        prompt: str,
        request: LLMRequest
    ) -> LLMResponse:
        """Call a specific LLM provider."""
        if provider == "openai":
            return await self._call_openai(prompt, request)
        elif provider == "anthropic":
            return await self._call_anthropic(prompt, request)
        elif provider == "google":
            return await self._call_google(prompt, request)
        elif provider == "huggingface":
            return await self._call_huggingface(prompt, request)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def _call_openai(self, prompt: str, request: LLMRequest) -> LLMResponse:
        """Call OpenAI API."""
        try:
            messages = []
            
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.openai_client.chat.completions.create(
                model=self.config.get('openai_model', 'gpt-4'),
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop_sequences or None,
                stream=request.stream
            )
            
            if request.stream:
                # Handle streaming response
                content_chunks = []
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content_chunks.append(chunk.choices[0].delta.content)
                
                content = ''.join(content_chunks)
                finish_reason = chunk.choices[0].finish_reason
                tokens_used = chunk.usage.total_tokens if hasattr(chunk, 'usage') else 0
            else:
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                tokens_used = response.usage.total_tokens
            
            return LLMResponse(
                content=content,
                model=response.model,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                safety_scores={},  # OpenAI doesn't provide safety scores
                constitutional_check=True,
                metadata={'provider': 'openai', 'model': response.model}
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _call_anthropic(self, prompt: str, request: LLMRequest) -> LLMResponse:
        """Call Anthropic API."""
        try:
            system_prompt = request.system_prompt or ""
            
            response = self.anthropic_client.messages.create(
                model=self.config.get('anthropic_model', 'claude-3-opus-20240229'),
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop_sequences=request.stop_sequences
            )
            
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            # Extract safety scores if available
            safety_scores = {}
            if hasattr(response, 'safety_ratings'):
                for rating in response.safety_ratings:
                    safety_scores[rating.category] = rating.level
            
            return LLMResponse(
                content=content,
                model=response.model,
                tokens_used=tokens_used,
                finish_reason="stop",
                safety_scores=safety_scores,
                constitutional_check=True,
                metadata={'provider': 'anthropic', 'model': response.model}
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def _call_google(self, prompt: str, request: LLMRequest) -> LLMResponse:
        """Call Google Generative AI API."""
        try:
            model_name = self.config.get('google_model', 'gemini-pro')
            model = self.google_client.GenerativeModel(model_name)
            
            # Combine system prompt and user prompt
            full_prompt = ""
            if request.system_prompt:
                full_prompt += f"{request.system_prompt}\n\n"
            full_prompt += prompt
            
            generation_config = {
                'temperature': request.temperature,
                'max_output_tokens': request.max_tokens,
                'stop_sequences': request.stop_sequences
            }
            
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            # Extract safety ratings
            safety_scores = {}
            if hasattr(response, 'prompt_feedback'):
                for rating in response.prompt_feedback.safety_ratings:
                    safety_scores[rating.category] = rating.probability
            
            # Get token counts if available
            tokens_used = 0
            if hasattr(response, 'usage_metadata'):
                tokens_used = (response.usage_metadata.prompt_token_count +
                             response.usage_metadata.candidates_token_count)
            
            return LLMResponse(
                content=response.text,
                model=model_name,
                tokens_used=tokens_used,
                finish_reason="stop",
                safety_scores=safety_scores,
                constitutional_check=True,
                metadata={'provider': 'google', 'model': model_name}
            )
            
        except Exception as e:
            logger.error(f"Google API error: {e}")
            raise
    
    async def _call_huggingface(self, prompt: str, request: LLMRequest) -> LLMResponse:
        """Call Hugging Face model."""
        try:
            # Load model if not already loaded
            if not hasattr(self, 'hf_model') or self.hf_model is None:
                model_name = self.hf_models[0] if self.hf_models else 'gpt2'
                
                self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                if self.hf_tokenizer.pad_token is None:
                    self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            
            # Tokenize input
            inputs = self.hf_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048 - request.max_tokens
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.hf_model.generate(
                    inputs.input_ids,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    do_sample=True,
                    pad_token_id=self.hf_tokenizer.pad_token_id,
                    eos_token_id=self.hf_tokenizer.eos_token_id
                )
            
            # Decode response
            response_text = self.hf_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Calculate tokens
            tokens_used = outputs.shape[1] - inputs.input_ids.shape[1]
            
            return LLMResponse(
                content=response_text,
                model=self.hf_models[0] if self.hf_models else 'huggingface_model',
                tokens_used=tokens_used,
                finish_reason="length",
                safety_scores={},
                constitutional_check=True,
                metadata={'provider': 'huggingface'}
            )
            
        except Exception as e:
            logger.error(f"Hugging Face error: {e}")
            raise
    
    def _select_provider(self, request: LLMRequest) -> Optional[str]:
        """Select the best provider for a request."""
        if not self.available_providers:
            return None
        
        # Simple provider selection logic
        # In production, this could consider cost, latency, capabilities, etc.
        
        # Check for provider preferences in request metadata
        if 'preferred_provider' in request.metadata:
            preferred = request.metadata['preferred_provider']
            if preferred in self.available_providers:
                return preferred
        
        # Check for model requirements
        if 'required_model' in request.metadata:
            required_model = request.metadata['required_model']
            
            if 'openai' in self.available_providers and 'gpt' in required_model.lower():
                return 'openai'
            elif 'anthropic' in self.available_providers and 'claude' in required_model.lower():
                return 'anthropic'
            elif 'google' in self.available_providers and 'gemini' in required_model.lower():
                return 'google'
        
        # Default to first available provider
        return self.available_providers[0]
    
    def _generate_cache_key(self, prompt: str, request: LLMRequest) -> str:
        """Generate cache key for a prompt."""
        key_data = {
            'prompt': prompt,
            'temperature': request.temperature,
            'max_tokens': request.max_tokens,
            'system_prompt': request.system_prompt,
            'guardrail_level': request.guardrail_level
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def _apply_rate_limit(self, provider: str):
        """Apply rate limiting for a provider."""
        if provider not in self.rate_limits:
            return
        
        rate_limit = self.rate_limits[provider]
        if 'requests_per_minute' in rate_limit:
            now = time.time()
            
            # Get recent requests for this provider
            if provider not in self.request_timestamps:
                self.request_timestamps[provider] = []
            
            # Remove timestamps older than 1 minute
            self.request_timestamps[provider] = [
                ts for ts in self.request_timestamps[provider]
                if now - ts < 60
            ]
            
            # Check if we're at the limit
            if len(self.request_timestamps[provider]) >= rate_limit['requests_per_minute']:
                # Wait until the oldest request is more than 1 minute old
                oldest = min(self.request_timestamps[provider])
                wait_time = 60 - (now - oldest)
                
                if wait_time > 0:
                    logger.debug(f"Rate limit reached for {provider}, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            
            # Add current timestamp
            self.request_timestamps[provider].append(now)
    
    def _update_metrics(self, response_time: float, tokens_used: int, success: bool):
        """Update performance metrics."""
        if success:
            self.metrics['tokens_used'] += tokens_used
            
            # Update average response time
            old_avg = self.metrics['average_response_time']
            n = self.metrics['successful_requests']
            self.metrics['average_response_time'] = (old_avg * (n - 1) + response_time) / n
    
    def set_rate_limit(self, provider: str, requests_per_minute: int):
        """Set rate limit for a provider."""
        if provider in self.available_providers:
            self.rate_limits[provider] = {
                'requests_per_minute': requests_per_minute
            }
            logger.info(f"Set rate limit for {provider}: {requests_per_minute} requests per minute")
    
    def clear_cache(self):
        """Clear the response cache."""
        self.response_cache.clear()
        logger.info("Response cache cleared")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()
    
    def get_provider_status(self) -> Dict[str, bool]:
        """Get status of all providers."""
        status = {}
        for provider in self.providers:
            status[provider] = provider in self.available_providers
        return status
