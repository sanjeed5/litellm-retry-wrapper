# LiteLLM Retry Wrapper

A robust wrapper for LiteLLM with built-in retry logic and rate limiting capabilities. This package helps you handle API failures gracefully and manage your API rate limits effectively.

## Features

- Automatic retry with exponential backoff for failed API calls
- Rate limiting to prevent API quota exhaustion
- Easy integration with existing LiteLLM implementations
- Configurable retry and rate limit parameters

## Installation

```bash
pip install litellm-retry-wrapper
```

Or with uv:

```bash
uv add litellm-retry-wrapper
```

## Usage

```python
from litellm_retry_wrapper import LiteLLMCaller

# Initialize with default settings
caller = LiteLLMCaller()

# Or with custom configuration
caller = LiteLLMCaller(
    model_name="gemini/gemini-2.0-flash",
    rpm=2000,
    max_retries=3,
    min_retry_wait=4,
    max_retry_wait=10
)

# Make API calls
response = caller.complete(
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=100
)
```

## Configuration

- `model_name`: The name of the model to use (default: "gemini/gemini-2.0-flash")
- `rpm`: Rate limit in requests per minute (default: 2000)
- `max_retries`: Maximum number of retry attempts (default: 3)
- `min_retry_wait`: Minimum wait time between retries in seconds (default: 4)
- `max_retry_wait`: Maximum wait time between retries in seconds (default: 10)

## Development

### Setup Development Environment

1. Clone the repository:
```bash
git clone https://github.com/sanjeed5/litellm-retry-wrapper.git
cd litellm-retry-wrapper
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LiteLLM](https://github.com/BerriAI/litellm)
- Uses [tenacity](https://github.com/jd/tenacity) for retry logic
- Uses [ratelimit](https://github.com/tomasbasham/ratelimit) for rate limiting
