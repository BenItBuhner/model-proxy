# Multi-Level Model Fallback Routing

This document describes the multi-level model fallback routing system that provides resilient LLM inference by automatically falling back through API keys, providers, and logical models.

## Overview

The fallback routing system implements three levels of fallback:

1. **API Key Fallback**: Try multiple API keys for the same provider
2. **Provider Fallback**: Try alternative providers for the same logical model
3. **Logical Model Fallback**: Fall back to entirely different logical models

This ensures high availability and resilience against provider outages, rate limits, and authentication issues.

## JSON Configuration Schema

Logical models are defined in JSON files under `config/models/<logical_model>.json`. Here's the complete schema:

```json
{
  "logical_name": "glm-4.6",
  "timeout_seconds": 60,
  "model_routings": [
    {
      "id": "primary",
      "wire_protocol": "openai",
      "provider": "cerebras",
      "model": "zai-glm-4.6",
      "base_url": "https://api.cerebras.ai/v1",
      "api_key_env": ["CEREBRAS_API_KEY", "CEREBRAS_API_KEY_BACKUP"],
      "timeout_seconds": 30
    },
    {
      "id": "secondary",
      "wire_protocol": "openai",
      "provider": "nahcrof",
      "model": "glm-4.6",
      "api_key_env": ["NAHCROF_API_KEY", "NAHCROF_API_KEY_BACKUP"]
    }
  ],
  "fallback_model_routings": ["glm-4.5", "qwen3-coder"]
}
```

### Field Descriptions

- `logical_name`: The logical model name (must match filename)
- `timeout_seconds`: Default timeout for all routes (optional, defaults to 60)
- `model_routings`: Array of provider routes to try (required)
  - `id`: Optional identifier for the route
  - `wire_protocol`: `"openai"` or `"anthropic"`
  - `provider`: Provider name (e.g., `"cerebras"`, `"nahcrof"`, `"openai"`)
  - `model`: Concrete model name to send to the provider
  - `base_url`: Override base URL for the provider (optional)
  - `api_key_env`: Array of environment variable names to try for API keys (ordered)
  - `timeout_seconds`: Route-specific timeout (optional)
- `fallback_model_routings`: Array of logical model names to fall back to (optional)

## Fallback Logic

### Level 1: API Key Fallback

For each route in `model_routings`, the system tries each API key in `api_key_env` order:

```json
{
  "provider": "cerebras",
  "api_key_env": ["CEREBRAS_KEY1", "CEREBRAS_KEY2", "CEREBRAS_KEY3"]
}
```

If `CEREBRAS_KEY1` fails, it tries `CEREBRAS_KEY2`, then `CEREBRAS_KEY3`.

### Level 2: Provider Fallback

After exhausting all API keys for the first route, it moves to the next route:

```json
"model_routings": [
  {"provider": "cerebras", "api_key_env": ["KEY1"]},  // Tried first
  {"provider": "nahcrof", "api_key_env": ["KEY2"]},   // Then this
  {"provider": "openai", "api_key_env": ["KEY3"]}     // Then this
]
```

### Level 3: Logical Model Fallback

If all routes for the primary logical model fail, it recursively tries fallback logical models:

```json
"fallback_model_routings": ["glm-4.5", "qwen3-coder"]
```

Each fallback model has its own complete routing configuration.

## Fallback-Worthy Failures

The system considers these errors as "fallback-worthy" (transient failures that should trigger retry):

- **HTTP Status Errors**: 4xx and 5xx status codes (except 401/403 auth errors)
- **Network Errors**: Connection refused, DNS resolution failures
- **Timeout Errors**: Client-side timeouts and server timeout responses

**Non-fallback-worthy errors** (permanent failures that fail immediately):
- Authentication errors (401/403)
- Invalid requests (400 with validation errors)

## Example Configurations

### GLM-4.6 with Multi-Provider Fallback

```json
{
  "logical_name": "glm-4.6",
  "timeout_seconds": 60,
  "model_routings": [
    {
      "id": "cerebras-primary",
      "wire_protocol": "openai",
      "provider": "cerebras",
      "model": "zai-glm-4.6",
      "api_key_env": ["CEREBRAS_API_KEY", "CEREBRAS_BACKUP_KEY"]
    },
    {
      "id": "nahcrof-secondary",
      "wire_protocol": "openai",
      "provider": "nahcrof",
      "model": "glm-4.6",
      "api_key_env": ["NAHCROF_API_KEY"]
    },
    {
      "id": "nahcrof-tertiary",
      "wire_protocol": "openai",
      "provider": "nahcrof",
      "model": "glm-4.6-turbo",
      "api_key_env": ["NAHCROF_API_KEY"]
    }
  ],
  "fallback_model_routings": ["glm-4.5", "qwen3-coder"]
}
```

### Anthropic-Compatible Model

```json
{
  "logical_name": "claude-3.5-sonnet",
  "model_routings": [
    {
      "wire_protocol": "anthropic",
      "provider": "anthropic",
      "model": "claude-3-5-sonnet-20241022",
      "api_key_env": ["ANTHROPIC_API_KEY", "ANTHROPIC_BACKUP_KEY"]
    }
  ]
}
```

## Testing with Fallback Tester

The `scripts/fallback_tester.py` script allows you to test fallback scenarios:

### List Available Models

```bash
python scripts/fallback_tester.py --list-models
```

### Basic Success Test

```bash
python scripts/fallback_tester.py glm-4.6 --scenario basic_success
```

### Test Different Scenarios

- `basic_success`: All routes work (no fallback needed)
- `api_key_fallback`: First API key fails, second succeeds
- `provider_fallback`: Primary provider fails, secondary succeeds
- `logical_model_fallback`: All primary routes fail, falls back to another model
- `timeout_fallback`: First route times out, second succeeds
- `all_fail`: All routes fail (shows error aggregation)

### Example Test Run

```bash
python scripts/fallback_tester.py glm-4.6 --scenario provider_fallback --verbose
```

Output:
```
============================================================
Testing fallback routing for 'glm-4.6' with scenario 'provider_fallback'
============================================================
[Attempt] glm-4.6 -> cerebras/zai-glm-4.6
[Attempt] glm-4.6 -> nahcrof/glm-4.6

✅ SUCCESS
   Duration: 0.15s
   Final provider: mock-nahrof-1
   Response: Mock response from nahrof (attempt 1)

📋 Call Log:
--------------------------------------------------------------------------------
  1. glm-4.6 -> cerebras/zai-glm-4.6 (openai) [FAILED]
  2. glm-4.6 -> nahcrof/glm-4.6 (openai) [SUCCESS]
```

## Integration with Existing Routers

The fallback routing is integrated transparently with existing OpenAI and Anthropic routers:

### OpenAI-Compatible Endpoints

- `/v1/chat/completions` automatically uses fallback routing if a logical model config exists
- Falls back to legacy routing if no config is found
- Maintains full OpenAI API compatibility

### Anthropic-Compatible Endpoints

- `/v1/messages` uses fallback routing for configured logical models
- Supports both OpenAI-compatible and native Anthropic providers
- Maintains full Anthropic API compatibility

## Environment Variables

Set these environment variables for API keys:

```bash
export CEREBRAS_API_KEY="your-cerebras-key"
export NAHCROF_API_KEY="your-nahcrof-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Monitoring and Logging

The system logs all routing attempts and fallbacks:

- Request logs show which logical model and concrete route was attempted
- Fallback events are logged with reasons
- Final successful routes are tracked for analytics

## Cycle Detection

The system prevents infinite loops by detecting cycles in fallback configurations:

- `model-a` → `model-b` → `model-a` (cycle detected, no fallback attempted)
- Each logical model tracks its "visited" status during resolution

## Error Aggregation

When all routes fail, the system provides detailed error information:

- Which logical models were attempted
- Which concrete routes failed and why
- Structured error information for debugging

## Adding New Logical Models

1. Create `config/models/<model-name>.json` with the routing configuration
2. Set appropriate environment variables for API keys
3. Test with the fallback tester
4. The model will automatically be available through existing endpoints

## Performance Considerations

- Configurations are cached in memory with file modification detection
- Failed routes are retried quickly without backoff (since they're different providers/keys)
- Successful routes are used immediately without additional latency
