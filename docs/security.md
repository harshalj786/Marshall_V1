# Security Policy & Implementation Guidelines (`SECURITY.md`)

This document outlines the practical security controls, data privacy standards, and operational safeguards for the **Marshall V2** agentic framework.

---

## 1. Secrets Management

*   **Zero Commit Policy:** Never commit raw API keys, OAuth secrets, database passwords, or private tokens (`.env`, `credentials.json`, `token.json`, `*.pem`) to version control.
*   **Environment Configuration:** All sensitive runtime keys must be loaded exclusively via environment variables using `.env` (gitignored by default).
*   **Compromise Protocol:**
    1. Immediately revoke the exposed token or API key in the respective provider dashboard (Google Cloud Console, OpenAI, Anthropic, etc.).
    2. Remove the secret from Git history (e.g., using `git-filter-repo` or `BFG Repo-Cleaner`).
    3. Generate a new secret and update local `.env` files.

---

## 2. Authentication

*   **OAuth 2.0 Integration (Gmail/Services):** Use PKCE-enabled OAuth 2.0 flows for external user services. Store access/refresh tokens locally in encrypted format or secure local token caches (`token.json`).
*   **Local Web Dashboard / API Auth:** If running a web interface, authenticate requests using short-lived **JWT (JSON Web Tokens)** stored in `HttpOnly`, `SameSite=Strict` cookies.
*   **Password Storage:** Hash all local admin or web dashboard user passwords using **Argon2id** (preferred) or **bcrypt** with a cost factor >= 12.
*   **Multi-Factor Authentication (MFA):** Enforce MFA on all upstream cloud accounts (Google Cloud, GitHub, LLM provider dashboards) linked to Marshall V2 services.

---

## 3. Authorization & Permissions

*   **Role-Based Access Control (RBAC):** Restrict system actions based on execution scope:
    *   `Admin`: Full agent configuration, tool enabling, and vector DB management.
    *   `Agent`: Restricted tool execution engine (cannot modify core codebase or system config).
*   **Principle of Least Privilege:**
    *   Google OAuth scopes must strictly request read-only access where possible (e.g., `https://www.googleapis.com/auth/gmail.readonly`) unless explicit action tools are enabled.
    *   Browser automation scripts operate in isolated context instances without access to ambient system cookies or local browser profiles.

---

## 4. LLM & Agent Security

*   **Prompt Injection Mitigation:** 
    *   Treat all untrusted external content (scraped website HTML, email bodies, YouTube transcripts) as untrusted user data.
    *   Pass untrusted input exclusively inside designated variable slots within isolated system prompts—never concatenate raw web data into system instructions.
*   **Tool Call Validation:**
    *   Validate all tool call arguments using strict **Pydantic** schemas before execution.
    *   Reject malformed or non-conforming LLM tool payloads prior to execution.
*   **Input Sanitization:** Sanitize and escape all user and web inputs before feeding them into embedding generators, ChromaDB queries, or sub-processes.
*   **File System Restrictions:**
    *   Limit agent read/write capabilities exclusively to designated directories (`db/`, `docs/`, `config/`).
    *   Explicitly forbid absolute file paths (`/etc`, `C:\Windows`) and directory traversal sequences (`../`).
*   **Command Execution Restrictions:**
    *   Strictly prohibit native sub-shell calls (`subprocess`, `os.system()`, `eval()`, `exec()`) within tool routines.
*   **Exception — the code tools** (`check_code` and `run_code` in `tools/code_tools.py`): these two tools, and only these, may start a subprocess (gcc/g++, or the program itself) and read a file given by path. The guards that stand in for the rules above:
    *   `run_code` is bound only in an **admin session** — unlocked at startup by the Argon2id-hashed admin password (§2) — and every run is confirmed by the user *outside the model*: a terminal `[y/N]`, or a spoken yes matched in code. The model can request a run; it can never supply the yes.
    *   **Guest sessions** get `check_code` with pasted code only. File paths are refused, and C/C++ may only `#include <...>` system headers: quoted, absolute and `..` includes (and `#embed`) are refused before the compiler sees the code, because compiler diagnostics quote source lines.
    *   The **admin** may pass paths, but only to `.py` `.c` `.h` `.cpp` `.cc` `.cxx` `.hpp` files — never `.env*` or the password hash — and the language argument cannot override the extension.
    *   Programs run in a fresh temporary directory with no stdin, a hard time limit, the whole process tree killed on timeout, and output capped before it reaches the model.
    *   Design: `docs/superpowers/specs/2026-11-10-code-check-and-run-design.md`.

---

## 5. Data Privacy & Storage

*   **Stored Data Scope:**
    *   ChromaDB vector indices (`db/`).
    *   Compressed user interaction context and key-value memory snippets.
    *   OAuth tokens and temporary scraped Markdown caches.
*   **Encryption Standard:**
    *   **In Transit:** All external HTTPS/WSS requests enforce TLS 1.3 (minimum TLS 1.2).
    *   **At Rest:** Local ChromaDB collections and state checkpointers containing user data must be stored on encrypted volumes (e.g., BitLocker, LUKS) or encrypted using AES-256 at the storage layer.
*   **Data Retention & Purging:**
    *   Short-term chat message histories clear automatically every 10 turns or upon reaching context window thresholds.
    *   Provide explicit maintenance scripts (`python main.py --clear-memory`) to purge persistent vector database history on demand.

---

## 6. Dependency Management

*   **Version Pinning:** Explicitly pin exact versions of all Python dependencies in `requirements.txt` (e.g., `langgraph==0.2.10`, `crawl4ai==0.3.5`).
*   **Vulnerability Scanning:** Run automated package dependency audits on every build/refactor using `pip-audit` or `safety`:
    ```bash
    pip-audit -r requirements.txt
    ```
*   **Update Cycle:** Routinely review and update core frameworks (LangGraph, Playwright, Ollama, ChromaDB) in controlled development branches before merging to main.

---

## 7. Logging & Telemetry

*   **Zero-Secret Logging:** Never output raw API keys, bearer tokens, or password hashes to terminal logs or persistent log files.
*   **PII & Token Masking:** Apply sanitization filters to mask sensitive information (e.g., email addresses, OAuth tokens, phone numbers) before emitting logs:
    ```text
    LOG: Fetching emails for user: h******@g*****.com | Token: [REDACTED]
    ```

---

## 8. Deployment & Environment Isolation

*   **HTTPS Only:** Enforce TLS/HTTPS for any web server endpoints or API hooks exposed by the application.
*   **Security Headers:** When serving web interfaces, enforce standard security headers:
    *   `Content-Security-Policy (CSP)`
    *   `X-Frame-Options: DENY`
    *   `Strict-Transport-Security (HSTS)`
    *   `X-Content-Type-Options: nosniff`
*   **Environment Separation:** Maintain strict separation between `DEVELOPMENT` (sandbox, mock tools, test databases) and `PRODUCTION` (live OAuth, active browser agents) execution modes.

---

## 9. Incident Response Plan

### Emergency Key Revocation Checklist
If a secret or credential leak occurs:
1. Run emergency token deletion script to clear active local sessions:
   ```bash
   rm -f token.json config/credentials.json .env