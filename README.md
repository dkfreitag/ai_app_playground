# AI App Playground

## Purpose

This is a sandbox where I can experiment with [Langgraph](https://docs.langchain.com/oss/python/langgraph/overview), [Pydantic AI](https://ai.pydantic.dev/models/overview/), and LLM's/Agents.

Rather than pay to make API calls to a model hosted by Anthropic/OpenAI/etc., I just use a local version of [gpt-oss:20b](https://ollama.com/library/gpt-oss:20b) with [Ollama](https://github.com/ollama/ollama).


## Installation

Download Ollama and install `gpt-oss:20b`
```
curl -fsSL https://ollama.com/install.sh | sh
ollama run gpt-oss:20b
```

Run a script:
```
uv run get_time_agent.py
```

Example output:
```
{'current_time': '2025-11-09 18:14:25.067771-05:00', 'timezone': 'America/New_York', 'utc_offset': '-05:00', 'month_name': 'November', 'month_emoji': '🍂', 'AM': False, 'PM': True}
```
