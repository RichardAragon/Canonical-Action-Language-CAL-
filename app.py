#!/usr/bin/env python3
# app.py — CAL (Canonical Action Language) single-file app
# Usage examples:
#   pip install transformers accelerate torch pydantic requests
#   python app.py --model google/gemma-3-270m --service jsonplaceholder --goal "Get posts for user 1"
#   python app.py --openapi ./openapi.json --base-url https://api.yourservice.com --goal "List the latest orders"

import argparse, json, os, re, sys
from typing import Any, Dict, List, Optional, Tuple

import requests

# --- Pydantic v2 preferred; v1 fallback is supported for dump() ---
try:
    from pydantic import BaseModel, Field
    PydanticV2 = True
except Exception:  # pydantic v1 fallback
    from pydantic import BaseModel  # type: ignore
    from pydantic.fields import Field  # type: ignore
    PydanticV2 = False

# --- Transformers (HF) ---
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception as e:
    print("Missing dependency: transformers. Install with `pip install transformers accelerate torch`.")
    raise

# =========================
#         CAL Core
# =========================

class CAL(BaseModel):
    service: str
    endpoint: str
    method: str = Field(pattern="^(GET|POST|PUT|PATCH|DELETE)$") if PydanticV2 else "GET"
    path_params: Dict[str, Any] = {} if not PydanticV2 else Field(default_factory=dict)
    query: Dict[str, Any]       = {} if not PydanticV2 else Field(default_factory=dict)
    headers: Dict[str, Any]     = {} if not PydanticV2 else Field(default_factory=dict)
    body: Dict[str, Any]        = {} if not PydanticV2 else Field(default_factory=dict)
    expect: Dict[str, Any]      = {"status": 200} if not PydanticV2 else Field(default_factory=lambda: {"status": 200})

def model_dump(obj) -> dict:
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()

class EndpointSig(BaseModel):
    method: str
    path: str
    required_query: List[str] = [] if not PydanticV2 else Field(default_factory=list)
    optional_query: List[str] = [] if not PydanticV2 else Field(default_factory=list)
    path_params: List[str]    = [] if not PydanticV2 else Field(default_factory=list)
    has_body: bool = False

class CompiledSpec(BaseModel):
    service: str
    base_url: str
    endpoints: List[EndpointSig]

class OpenAPISubsetCompiler:
    """
    Compile a (possibly large) OpenAPI dict into a tiny endpoint signature list this app uses.
    Supports parameters (query/path) and detects if a requestBody exists.
    """
    @staticmethod
    def compile(service: str, base_url: str, openapi_like: Dict[str, Any]) -> CompiledSpec:
        eps: List[EndpointSig] = []
        for path, methods in openapi_like.get("paths", {}).items():
            for m, spec in (methods or {}).items():
                mu = str(m).upper()
                if mu not in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
                    continue
                req_q, opt_q, path_params = [], [], []
                for param in spec.get("parameters", []) or []:
                    where = param.get("in")
                    name = param.get("name")
                    required = bool(param.get("required", False))
                    if where == "query":
                        (req_q if required else opt_q).append(name)
                    elif where == "path":
                        path_params.append(name)
                has_body = "requestBody" in spec
                eps.append(EndpointSig(method=mu, path=path,
                                       required_query=req_q, optional_query=opt_q,
                                       path_params=path_params, has_body=has_body))
        return CompiledSpec(service=service, base_url=base_url, endpoints=eps)

# ---------- Demo catalog (keeps app usable out of the box) ----------
def jsonplaceholder_openapi_subset() -> Dict[str, Any]:
    return {
        "paths": {
            "/posts": {"get": {"parameters": [
                {"in": "query", "name": "userId", "required": False}
            ]}},
            "/posts/{id}": {"get": {"parameters": [
                {"in": "path", "name": "id", "required": True}
            ]}},
            "/users": {"get": {"parameters": []}},
            "/users/{id}": {"get": {"parameters": [
                {"in": "path", "name": "id", "required": True}
            ]}},
            "/comments": {"get": {"parameters": [
                {"in": "query", "name": "postId", "required": False}
            ]}},
        }
    }

# =========================
#          Utils
# =========================

def robust_extract_json(text: str) -> Optional[dict]:
    # Prefer explicit markers
    if "BEGIN_CAL" in text and "END_CAL" in text:
        inner = text.split("BEGIN_CAL", 1)[1].split("END_CAL", 1)[0]
        try:
            return json.loads(inner.strip())
        except Exception:
            pass
    # Fallback: search for the last balanced JSON object
    last = text.rfind("}")
    while last != -1:
        start = text.rfind("{", 0, last)
        while start != -1:
            chunk = text[start:last+1]
            bal = 0
            ok = True
            for ch in chunk:
                if ch == "{": bal += 1
                elif ch == "}":
                    bal -= 1
                    if bal < 0:
                        ok = False; break
            if ok and bal == 0:
                try:
                    return json.loads(chunk)
                except Exception:
                    pass
            start = text.rfind("{", 0, start)
        last = text.rfind("}", 0, last)
    return None

def extract_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text, flags=re.I)
    return int(m.group(1)) if m else None

# =========================
#         LLM
# =========================

class LLM:
    def __init__(self, model_name: str, hf_token: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=self.hf_token,
            device_map="auto",
        )
        self.device = device or ("cuda" if self.model.device.type == "cuda" else "cpu")

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.model.device:
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,         # deterministic → helps JSON-only
                temperature=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text[len(prompt):].strip()

# =========================
#     Planner / Executor
# =========================

PLANNER_SYSTEM = """You are an API planner.
Emit exactly ONE Canonical Action Language (CAL) JSON.
STRICT FORMAT:
Return ONLY between these markers:
BEGIN_CAL
{ ... JSON only ... }
END_CAL
No other text before/after. Schema keys: service, endpoint, method, path_params, query, headers, body, expect.
Rules:
- Use only the listed endpoints.
- Fill required path params.
- Prefer GET if unsure.
- If goal mentions 'user N', set query.userId = N when endpoint supports it.
- Use empty objects for headers/body when not needed.
- Default expect.status = 200.
"""

def few_shot_examples(service: str) -> str:
    ex1 = {
        "service": service, "endpoint": "/users", "method": "GET",
        "path_params": {}, "query": {}, "headers": {}, "body": {},
        "expect": {"status": 200}
    }
    ex2 = {
        "service": service, "endpoint": "/posts", "method": "GET",
        "path_params": {}, "query": {"userId": 1}, "headers": {}, "body": {},
        "expect": {"status": 200}
    }
    return f"BEGIN_CAL\n{json.dumps(ex1)}\nEND_CAL\nBEGIN_CAL\n{json.dumps(ex2)}\nEND_CAL"

class Planner:
    def __init__(self, llm: LLM):
        self.llm = llm

    def plan(self, user_goal: str, compiled: CompiledSpec) -> Tuple[Optional[CAL], str]:
        endpoints_desc = "\n".join(
            [f"- {e.method} {e.path} | req_q={e.required_query} opt_q={e.optional_query} path={e.path_params}"
             for e in compiled.endpoints]
        )
        uid = extract_int(r"user\s*(\d+)", user_goal)
        hint = f"Detected userId={uid} from goal." if uid is not None else "No userId detected."

        prompt = (
            f"{PLANNER_SYSTEM}\n\n"
            f"Service: {compiled.service}\nBase URL: {compiled.base_url}\n"
            f"Available endpoints:\n{endpoints_desc}\n\n"
            f"Examples:\n{few_shot_examples(compiled.service)}\n\n"
            f"User goal: {user_goal}\n{hint}\n"
            f"Emit CAL JSON now between markers:"
        )
        out = self.llm.generate(prompt, max_new_tokens=256, temperature=0.0)
        data = robust_extract_json(out)
        if data is None:
            return None, out
        try:
            if uid is not None and data.get("endpoint") == "/posts":
                data.setdefault("query", {})
                data["query"].setdefault("userId", uid)
            cal = CAL(**data)
            return cal, out
        except Exception:
            return None, out

class Executor:
    def __init__(self, compiled_specs: Dict[str, CompiledSpec]):
        self.specs = compiled_specs

    def _format_url(self, service: str, endpoint: str, path_params: Dict[str, Any]) -> str:
        base = self.specs[service].base_url
        path = endpoint
        for k, v in (path_params or {}).items():
            path = path.replace("{"+str(k)+"}", str(v))
        return base + path

    def call(self, cal: CAL) -> Tuple[int, Any, str]:
        from requests.models import PreparedRequest
        url = self._format_url(cal.service, cal.endpoint, cal.path_params)
        pr = PreparedRequest()
        pr.prepare_url(url, cal.query if cal.query else None)
        full_url = pr.url
        try:
            resp = requests.request(
                cal.method.upper(), url,
                params=cal.query or None,
                headers=cal.headers or None,
                json=(cal.body or None),
                timeout=30
            )
            status = resp.status_code
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            return status, body, full_url
        except Exception as e:
            return -1, {"error": str(e)}, full_url

# =========================
#     Critic / Orchestrator
# =========================

class CriticRepair:
    def __init__(self, llm: LLM):
        self.llm = llm

    def validate(self, cal: CAL, status: int, body: Any) -> Tuple[bool, str]:
        exp = cal.expect.get("status", 200)
        if status != exp:
            return False, f"Expected status {exp}, got {status}."
        # Tiny domain-specific check (example): if filtering posts by userId, ensure they match.
        if cal.endpoint == "/posts" and isinstance(body, list) and "userId" in (cal.query or {}):
            uid = cal.query["userId"]
            if not all(isinstance(x, dict) and x.get("userId") == uid for x in body):
                return False, f"Validation failed: items do not match userId={uid}."
        return True, "OK"

    def repair(self, cal_json: str, error: str, endpoints_desc: str) -> Optional[CAL]:
        prompt = f"""You are an API repair assistant.
We attempted a call but it failed with error:
{error}

Available endpoints:
{endpoints_desc}

Return ONLY corrected CAL JSON between markers:
BEGIN_CAL
{{ ... }}
END_CAL

Previous CAL:
{cal_json}
"""
        out = self.llm.generate(prompt, max_new_tokens=256, temperature=0.0)
        data = robust_extract_json(out)
        if not data: return None
        try:
            return CAL(**data)
        except Exception:
            return None

class Orchestrator:
    def __init__(self, llm: LLM, compiled_specs: Dict[str, CompiledSpec]):
        self.planner = Planner(llm)
        self.executor = Executor(compiled_specs)
        self.critic = CriticRepair(llm)
        self.compiled_specs = compiled_specs

    def endpoints_desc(self, service: str) -> str:
        comp = self.compiled_specs[service]
        return "\n".join(
            [f"- {e.method} {e.path} | req_q={e.required_query} opt_q={e.optional_query} path={e.path_params}"
             for e in comp.endpoints]
        )

    def run(self, user_goal: str, service: str, max_repairs: int = 2) -> Dict[str, Any]:
        comp = self.compiled_specs[service]
        cal, raw = self.planner.plan(user_goal, comp)
        trace: Dict[str, Any] = {
            "goal": user_goal,
            "service": service,
            "plan_cal": model_dump(cal) if cal else None,
            "steps": []
        }
        if not cal:
            trace["error"] = "Planner failed to produce valid CAL"
            trace["plan_raw"] = raw
            return trace

        for attempt in range(max_repairs + 1):
            status, body, full_url = self.executor.call(cal)
            ok, reason = self.critic.validate(cal, status, body)
            step = {
                "attempt": attempt,
                "cal": model_dump(cal),
                "url": full_url,
                "status": status,
                "ok": ok,
                "reason": reason,
                "preview": (str(body)[:300] if not isinstance(body, (dict, list)) else str(body)[:300])
            }
            trace["steps"].append(step)
            if ok:
                trace["result"] = {
                    "status": status,
                    "url": full_url,
                    "body_preview": (str(body)[:500] if not isinstance(body, (dict, list)) else str(body)[:500])
                }
                return trace

            repaired = self.critic.repair(json.dumps(model_dump(cal)), f"status={status} reason={reason}",
                                          self.endpoints_desc(service))
            if not repaired:
                trace["error"] = "Repair failed to produce valid CAL"
                return trace
            cal = repaired

        trace["error"] = "Exceeded repair attempts"
        return trace

# =========================
#        Entrypoint
# =========================

def load_openapi(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_compiled(service: str, base_url: str, openapi: Dict[str, Any]) -> Dict[str, CompiledSpec]:
    return {service: OpenAPISubsetCompiler.compile(service, base_url, openapi)}

def default_catalog() -> Dict[str, CompiledSpec]:
    service = "jsonplaceholder"
    base_url = "https://jsonplaceholder.typicode.com"
    return {service: OpenAPISubsetCompiler.compile(service, base_url, jsonplaceholder_openapi_subset())}

def main():
    ap = argparse.ArgumentParser(description="CAL (Canonical Action Language) app")
    ap.add_argument("--model", default="google/gemma-3-270m", help="Hugging Face model id")
    ap.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Hugging Face token (for gated models)")
    ap.add_argument("--goal", required=True, help="User goal, e.g. 'Get posts for user 1'")
    ap.add_argument("--service", default="jsonplaceholder", help="Service name (logical)")
    ap.add_argument("--openapi", help="Path to an openapi.json (if not using built-in demo)")
    ap.add_argument("--base-url", help="Required with --openapi: base URL for the API")
    ap.add_argument("--max-repairs", type=int, default=2, help="Max repair attempts on failure")
    args = ap.parse_args()

    # Build compiled specs (either from provided OpenAPI or demo catalog)
    if args.openapi:
        if not args.base_url:
            print("--base-url is required when using --openapi", file=sys.stderr)
            sys.exit(2)
        openapi = load_openapi(args.openapi)
        compiled_specs = build_compiled(args.service, args.base_url, openapi)
    else:
        compiled_specs = default_catalog()

    # LLM + run
    llm = LLM(args.model, hf_token=args.hf_token)
    orch = Orchestrator(llm, compiled_specs)
    trace = orch.run(args.goal, args.service, max_repairs=args.max_repairs)
    print(json.dumps(trace, indent=2))

if __name__ == "__main__":
    main()
