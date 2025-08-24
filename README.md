# CAL — Canonical Action Language
*A tiny, consistent JSON “language” that makes AI speak API.*

> **Tagline:** Compile API specs into a uniform action language. Let the model plan in CAL, execute, validate, and auto-repair — so it generalizes across APIs with little data.

---

## Why
- **Every API is different.** Paths, params, auth, pagination — variations explode.
- **Wrappers don’t scale.** Writing a custom tool per API is brittle and hard to maintain.
- **Data hunger.** Collecting thousands of per-API examples is expensive and still fragile.

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

