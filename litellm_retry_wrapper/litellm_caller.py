import os
from litellm import completion
import tenacity
from ratelimit import limits, sleep_and_retry
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import threading
from collections import deque
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self.lock = threading.Lock()

    def _clean_old_requests(self):
        now = datetime.now()
        while self.requests and (now - self.requests[0]) > timedelta(seconds=self.window_seconds):
            self.requests.popleft()

    def try_acquire(self) -> bool:
        with self.lock:
            now = datetime.now()
            self._clean_old_requests()
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False

    def wait_if_needed(self):
        while not self.try_acquire():
            time.sleep(0.1)  # Sleep briefly before retrying

class LiteLLMCaller:
    # Default rate limits for common providers
    DEFAULT_RATE_LIMITS = {
        "gpt-3.5-turbo": {"rpm": 500},  # OpenAI default tier
        "gpt-4": {"rpm": 200},          # OpenAI default tier
        "gemini-pro": {"rpm": 600},     # Google Cloud default
        "claude-2": {"rpm": 400},       # Anthropic estimate
        "gemini/gemini-2.0-flash": {"rpm": 2000},
    }

    def __init__(
        self,
        model_name: str = "gemini/gemini-2.0-flash",
        rpm: Optional[int] = None,
        max_retries: int = 3,
        min_retry_wait: int = 4,
        max_retry_wait: int = 10
    ):
        """
        Initialize the LiteLLM caller with configurable parameters
        
        Args:
            model_name: The name of the model to use
            rpm: Rate limit (requests per minute). If None, uses default for model
            max_retries: Maximum number of retry attempts
            min_retry_wait: Minimum wait time between retries (seconds)
            max_retry_wait: Maximum wait time between retries (seconds)
        """
        self.model_name = model_name
        
        # Set appropriate rate limit based on model
        if rpm is None:
            # First try exact match
            if model_name in self.DEFAULT_RATE_LIMITS:
                rpm = self.DEFAULT_RATE_LIMITS[model_name]["rpm"]
            else:
                # Then try partial match
                base_model = model_name.split('/')[-1]
                for model_prefix, limits in self.DEFAULT_RATE_LIMITS.items():
                    if model_prefix in base_model:
                        rpm = limits["rpm"]
                        break
                rpm = rpm or 100  # Conservative default if no match

        self.rpm = rpm
        self.max_retries = max_retries
        self.min_retry_wait = min_retry_wait
        self.max_retry_wait = max_retry_wait
        
        # Initialize sliding window rate limiter
        self.rate_limiter = SlidingWindowRateLimiter(
            max_requests=self.rpm,
            window_seconds=60
        )
        
    def _rate_limited_completion(self, *args, **kwargs):
        """Make a rate-limited completion call"""
        self.rate_limiter.wait_if_needed()
        return completion(*args, **kwargs)
    
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.info(f"Retrying after {retry_state.next_action.sleep} seconds...")
    )
    def _make_completion_with_retry(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ):
        """
        Make a completion call with retry logic and rate limiting
        """
        try:
            response = self._rate_limited_completion(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"Error making completion call: {str(e)}")
            raise

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Public method to make a completion call
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in the response
            **kwargs: Additional arguments to pass to the completion call
            
        Returns:
            The completion response
            
        Raises:
            Exception: If the completion call fails after all retries
        """
        try:
            completion_kwargs = {
                'temperature': temperature,
                **kwargs
            }
            if max_tokens is not None:
                completion_kwargs['max_tokens'] = max_tokens
                
            response = self._make_completion_with_retry(
                messages=messages,
                **completion_kwargs
            )
            return response
        except Exception as e:
            logger.error(f"Failed to get completion after retries: {str(e)}")
            raise

# Example usage
def get_example_usage():
    # Initialize the caller
    llm_caller = LiteLLMCaller()
    
    # Example messages
    messages = [
        {
            "role": "user",
            "content": "Write a short poem about artificial intelligence."
        }
    ]
    
    try:
        # Make the API call
        response = llm_caller.complete(
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        
        # Return the response
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Failed to get completion: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        result = get_example_usage()
        print("Response:", result)
    except Exception as e:
        print(f"Error: {str(e)}")

