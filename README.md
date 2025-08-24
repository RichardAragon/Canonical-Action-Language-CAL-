# CAL — Canonical Action Language
*A tiny, consistent JSON “language” that makes AI speak API.*

> **Tagline:** Compile API specs into a uniform action language. Let the model plan in CAL, execute, validate, and auto‑repair — so it generalizes across APIs with little data.

---

## Why
- **Every API is different.** Paths, params, auth, pagination — variations explode.
- **Wrappers don’t scale.** Writing a custom tool per API is brittle and hard to maintain.
- **Data hunger.** Collecting thousands of per‑API examples is expensive and still fragile.

**CAL reframes APIs as a language, not a pile of functions.** If a model can write English or Python, it can write CAL.

---

## What is CAL?
CAL is a tiny JSON schema for *one API call*. A runtime executes that JSON; a validator checks results; and a critic proposes minimal fixes when something fails.

**Example (goal: “Get posts for user 1”):**
```json
{
  "service": "jsonplaceholder",
  "endpoint": "/posts",
  "method": "GET",
  "path_params": {},
  "query": { "userId": 1 },
  "headers": {},
  "body": {},
  "expect": { "status": 200 }
}
```

**Core fields**
- `service` — logical name for the API surface
- `endpoint` — path template (e.g., `/users/{id}`)
- `method` — `GET | POST | PUT | PATCH | DELETE`
- `path_params` — values for placeholders in the path
- `query` — querystring parameters
- `headers` — request headers (incl. auth)
- `body` — JSON request body
- `expect` — lightweight checks (e.g., `{ "status": 200 }`)

---

## How it works
1. **Compile**: Take an OpenAPI/GraphQL/proto spec and compile it into a small list of endpoint signatures & constraints.
2. **Plan**: The LLM emits **one CAL JSON** for the user’s goal (deterministic JSON-only output).
3. **Execute**: The runtime turns CAL into real HTTP/library calls.
4. **Validate & Repair**: If the call fails or violates constraints, a critic proposes a corrected CAL (guided by the spec + error text). Try again.

This **plan → execute → repair** loop means you need far less per‑API supervision.

---

## Quickstart

### 1) Install
```bash
pip install transformers accelerate torch pydantic requests
```

> If your chosen model is gated on Hugging Face, create a token at https://huggingface.co/settings/tokens and pass it with `--hf-token` or set `export HF_TOKEN=...`.

### 2) Run the demo (uses JSONPlaceholder)
```bash
python app.py \
  --model google/gemma-3-270m \
  --service jsonplaceholder \
  --goal "Get posts for user 1" \
  --hf-token $HF_TOKEN
```

### 3) Use your own API
```bash
python app.py \
  --model google/gemma-3-270m \
  --openapi ./openapi.json \
  --base-url https://api.example.com \
  --service example \
  --goal "List the latest orders" \
  --hf-token $HF_TOKEN
```

**Flags**
- `--model` — HF model id (default: `google/gemma-3-270m`)
- `--hf-token` — Hugging Face token for gated models (or use env `HF_TOKEN`)
- `--goal` — natural language goal (required)
- `--service` — logical service name (required when `--openapi` is used)
- `--openapi` — path to an OpenAPI JSON file (optional; otherwise demo catalog is used)
- `--base-url` — base URL for the API (required with `--openapi`)
- `--max-repairs` — max repair attempts (default: 2)

**Output**
The CLI prints a JSON trace containing:
- `plan_cal` — the final CAL JSON produced by the planner
- `steps` — attempts with URL, status, and a preview of the response
- `result` — final status, URL, and response preview

---

## Project layout
```
.
├── app.py           # Single-file CAL app (compile → plan → execute → repair)
└── README.md        # This file
```

---

## Architecture (in this repo)
- **Spec Compiler** — parses OpenAPI into minimal endpoint signatures (method, path, params, body flag).
- **Planner** — prompts the LLM to emit **exactly one** CAL JSON using explicit BEGIN/END markers and deterministic decoding.
- **Executor** — builds the request (fills `{id}`, adds query/body/headers), calls the API, and logs the full URL and response.
- **Critic / Repair** — when validation fails, it proposes a minimal CAL edit using the spec + error text.

> The validator currently checks status codes and (as an example) ensures `/posts?userId=N` really returns only `userId=N`. You can extend validators per endpoint/domain, or validate with a JSON Schema for the response body.

---

## CAL vs Model Context Protocol (MCP)
**MCP is plumbing; CAL is language + planning.**

- **MCP (Model Context Protocol)** standardizes how models connect to *tool servers* and resources (list tools, call them, stream results).
- **CAL** standardizes how a model *expresses API calls* (tiny JSON DSL + validate/repair loop) so it can generalize across many APIs with minimal per-API data.

They’re complementary: you can **plan in CAL** and **execute via MCP** (CAL → adapter → MCP tool calls).

---

## Security & governance
- **Least privilege:** Whitelist allowed endpoints/methods and block destructive ops by default.
- **Secrets management:** Inject tokens from a secure store (don’t log raw secrets).
- **Idempotency & retries:** Use idempotency headers with exponential backoff.
- **Human-in-the-loop:** Require confirmations for high-risk writes.
- **Auditability:** Keep immutable logs of CAL + responses for audits.

---

## Roadmap (ideas)
- Constrained decoding under a spec-derived grammar
- OAuth2/HMAC adapters with auto-refresh
- Pagination/backoff helpers + join/merge utilities
- Response validation with JSON Schema / Pydantic models
- Multi-step plans (CAL sequences) with dependency passing
- MCP adapter (CAL→MCP tool calls) and MCP server that hosts CAL execution
- Benchmarks: generalization across N unseen APIs with near-zero per-API data

---

## Troubleshooting
- **401 / gated model** — Pass `--hf-token` or set `HF_TOKEN`. Ensure your token has “read” access.
- **Invalid JSON output** — The planner enforces `BEGIN_CAL`/`END_CAL`. If you swap models, keep deterministic decoding (no sampling).
- **Spec quirks** — Some business rules live outside the spec; add small validators or a few curated examples.
- **Networking** — Ensure you can reach the API from your environment; proxies and CORS don’t apply to this CLI.

---

## Contributing
PRs and issues welcome! Good first contributions:
- Add response validators for common patterns
- Add OAuth2/HMAC auth adapters
- Improve the OpenAPI compiler’s coverage (enums, oneOf/allOf, etc.)
- Add new demo APIs to the built-in catalog

---

## License
MIT
