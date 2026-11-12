# Marshall V2 — Agent & Tool Architecture Specifications

## Overview
This document specifies the exact functional requirements, signatures, dependencies, and execution flows for all tool modules integrated into the Marshall V2 LangGraph engine (`tools/`).

---

## 1. Gmail Triage Tool (`tools/gmail_tool.py`)

### Tech Stack & Prerequisites
- **Library:** `google-api-python-client`, `google-auth-oauthlib`, `google-auth-httplib2`
- **Authentication:** OAuth2 Client Credentials (`credentials.json`), cached token in `token.json`.
- **Parser:** Structured LLM extraction via Pydantic (`EmailTriageBatch`).

### Function Specifications
```python
@tool
def fetch_and_triage_emails(max_results: int = 15, query: str = "is:unread") -> str:
    """
    Fetches recent emails matching the query, runs them through the LLM triage classifier,
    marks ignorable marketing/spam emails as read, and returns a categorized summary digest.
    """