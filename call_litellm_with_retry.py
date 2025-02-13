import os
from litellm import completion
import tenacity
from ratelimit import limits, sleep_and_retry
import logging
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiteLLMCaller:
    def __init__(
        self,
        model_name: str = "gemini/gemini-2.0-flash",
        rpm: int = 2000,
        max_retries: int = 3,
        min_retry_wait: int = 4,
        max_retry_wait: int = 10
    ):
        """
        Initialize the LiteLLM caller with configurable parameters
        
        Args:
            model_name: The name of the model to use
            rpm: Rate limit (requests per minute)
            max_retries: Maximum number of retry attempts
            min_retry_wait: Minimum wait time between retries (seconds)
            max_retry_wait: Maximum wait time between retries (seconds)
        """
        self.model_name = model_name
        self.rpm = rpm
        self.max_retries = max_retries
        self.min_retry_wait = min_retry_wait
        self.max_retry_wait = max_retry_wait
        
        # Create rate limited completion function
        self.rate_limited_completion = self._create_rate_limited_completion()
        
    def _create_rate_limited_completion(self):
        """Create a rate-limited version of the completion function"""
        @sleep_and_retry
        @limits(calls=self.rpm, period=60)
        def _rate_limited_completion(*args, **kwargs):
            return completion(*args, **kwargs)
        return _rate_limited_completion
    
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
            response = self.rate_limited_completion(
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

