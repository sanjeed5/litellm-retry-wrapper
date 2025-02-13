import pytest
from unittest.mock import patch, MagicMock
import time
from datetime import datetime, timedelta
from litellm_retry_wrapper import LiteLLMCaller
from litellm_retry_wrapper.litellm_caller import SlidingWindowRateLimiter

def test_rate_limiter_basic():
    """Test basic rate limiting functionality"""
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=1)
    
    # First two requests should succeed immediately
    assert limiter.try_acquire() == True
    assert limiter.try_acquire() == True
    
    # Third request should fail
    assert limiter.try_acquire() == False
    
    # After waiting, should succeed again
    time.sleep(1.1)
    assert limiter.try_acquire() == True

def test_rate_limiter_sliding_window():
    """Test sliding window behavior"""
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=2)
    
    # Make initial requests
    assert limiter.try_acquire() == True
    time.sleep(1)  # Wait 1 second
    assert limiter.try_acquire() == True
    
    # Should fail immediately after
    assert limiter.try_acquire() == False
    
    # After 1 more second, first request should have expired
    time.sleep(1)
    assert limiter.try_acquire() == True

@pytest.fixture
def mock_completion():
    with patch('litellm_retry_wrapper.litellm_caller.completion') as mock:
        mock.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response"))]
        )
        yield mock

def test_caller_initialization():
    """Test LiteLLMCaller initialization with different models"""
    # Test default model
    caller = LiteLLMCaller()
    assert caller.model_name == "gemini/gemini-2.0-flash"
    assert caller.rpm == 2000

    # Test custom RPM
    caller = LiteLLMCaller(rpm=100)
    assert caller.rpm == 100

    # Test known model
    caller = LiteLLMCaller(model_name="gpt-3.5-turbo")
    assert caller.rpm == 500

    # Test unknown model (should use conservative default)
    caller = LiteLLMCaller(model_name="unknown-model")
    assert caller.rpm == 100

def test_retry_logic(mock_completion):
    """Test retry behavior on failures"""
    mock_completion.side_effect = [
        Exception("First failure"),
        Exception("Second failure"),
        MagicMock(choices=[MagicMock(message=MagicMock(content="Success"))])
    ]
    
    caller = LiteLLMCaller()
    response = caller.complete(messages=[{"role": "user", "content": "test"}])
    
    assert mock_completion.call_count == 3
    assert response.choices[0].message.content == "Success"

def test_rate_limiting_integration(mock_completion):
    """Test rate limiting integration with completion calls"""
    caller = LiteLLMCaller(rpm=2)  # Set very low RPM for testing
    start_time = time.time()
    
    # Make 3 calls
    for _ in range(3):
        caller.complete(messages=[{"role": "user", "content": "test"}])
    
    duration = time.time() - start_time
    # Should take at least 60 seconds for the third call due to rate limiting
    assert duration >= 60

def test_completion_parameters(mock_completion):
    """Test that completion parameters are passed correctly"""
    caller = LiteLLMCaller()
    caller.complete(
        messages=[{"role": "user", "content": "test"}],
        temperature=0.5,
        max_tokens=100
    )
    
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args[1]
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["max_tokens"] == 100
    assert call_kwargs["model"] == "gemini/gemini-2.0-flash" 