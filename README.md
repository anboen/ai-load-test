# vLLM OpenAI-style client

This repository contains a small Python CLI that uses the official
`openai` Python library pointed at a vLLM server's OpenAI-compatible API
endpoint (via `api_base`). It's useful when your vLLM server implements
the OpenAI chat completions API surface.

Files

- `scripts/vllm_openai_client.py` — CLI that calls `openai.ChatCompletion.create`.
- `requirements.txt` — minimal dependency list (`openai`).

Quick start (PowerShell)

1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

2. Run a synchronous chat request

```powershell
python scripts/vllm_openai_client.py --api-base http://127.0.0.1:8000/v1 \
  --model my-model --message "Write a two-line haiku about fog"
```

3. Run streaming (if server supports it)

```powershell
python scripts/vllm_openai_client.py --api-base http://127.0.0.1:8000/v1 \
  --model my-model --message "Tell a short story" --stream
```

Environment

- Set `VLLM_API_BASE` to change the default base URL.
- Provide an API key with `--api-key` or set `VLLM_API_KEY` / `OPENAI_API_KEY`.

Notes

- The script is best-effort for both streaming and non-streaming flows and
  extracts assistant content from OpenAI-compatible responses. If your
  vLLM server uses a different request/response shape, pass a different
  `--api-base` or adjust the call in `scripts/vllm_openai_client.py`.
