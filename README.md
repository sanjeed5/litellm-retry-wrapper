# LiteLLM Retry Wrapper

A robust Python wrapper for LiteLLM that provides retry mechanisms, rate limiting, and error handling for LLM API calls.

## Features

- 🔄 Automatic retry mechanism with exponential backoff
- ⏱️ Built-in rate limiting
- 🎯 Configurable parameters for retries and rate limits
- 📝 Comprehensive logging
- 🛡️ Error handling and exception management
- 🔧 Easy to customize and extend

## Installation

```bash
uv pip install litellm-retry-wrapper
```

## Quick Start

```python
from call_litellm_with_retry import LiteLLMCaller

# Initialize the caller
llm_caller = LiteLLMCaller(
    model_name="gemini/gemini-2.0-flash",
    rpm=2000,
    max_retries=3
)

# Prepare your messages
messages = [
    {
        "role": "user",
        "content": "Write a short poem about artificial intelligence."
    }
]

# Make the API call
response = llm_caller.complete(
    messages=messages,
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)
```

## Configuration

The `LiteLLMCaller` class accepts the following parameters:

- `model_name`: The name of the LLM model to use (default: "gemini/gemini-2.0-flash")
- `rpm`: Rate limit in requests per minute (default: 2000)
- `max_retries`: Maximum number of retry attempts (default: 3)
- `min_retry_wait`: Minimum wait time between retries in seconds (default: 4)
- `max_retry_wait`: Maximum wait time between retries in seconds (default: 10)

## Environment Variables

Create a `.env` file with your API keys:

```env
GEMINI_API_KEY=your_api_key_here
```

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/sanjeed5/litellm-retry-wrapper.git
cd litellm-retry-wrapper

# Install dependencies
uv venv
source .venv/bin/activate
uv sync
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LiteLLM](https://github.com/BerriAI/litellm)
- Uses [tenacity](https://github.com/jd/tenacity) for retry logic
- Uses [ratelimit](https://github.com/tomasbasham/ratelimit) for rate limiting
