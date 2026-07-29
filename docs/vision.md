# Marshall V2 — Vision & Core Philosophy

## Executive Summary
Marshall V2 represents the evolution of Marshall from a passive, query-driven Retrieval-Augmented Generation (RAG) system into an active, stateful, and autonomous AI agent. Designed as an intelligent context-aware companion, Marshall V2 blends local privacy-first models with powerful real-world execution capabilities.

---

## The Evolutionary Leap: V1 vs. V2

| Feature | Marshall V1 | Marshall V2 |
| :--- | :--- | :--- |
| **Architecture** | Linear LangChain `Runnable` Pipeline | Cyclic State Machine via **LangGraph** |
| **Role** | Knowledge Retrieval & Contextual Q&A | Autonomous Task Execution & Intelligence Triage |
| **Action Capacity** | Passive Output Generation | Active Tool Invocation (Gmail, Browser, Scraper) |
| **Safety Control** | Static Prompt Guardrails | Native **Human-in-the-Loop (HITL)** Interrupts |
| **Execution Loop** | Single-shot / Rolling Window REPL | Multi-turn Reasoning & Tool Re-evaluation Loops |

---

## Core Pillars & Objectives

1. **Intelligent Daily Triage:**
   Provide high-signal, zero-noise email processing that categorizes, filters, and highlights urgent actions automatically.

2. **Autonomous Web Execution:**
   Handle multi-step online tasks (reservations, bookings, food ordering) through natural language while maintaining human financial oversight.

3. **Dynamic Market & Social Intelligence:**
   Scrape dynamic, web-based social trends and raw data to output structured sentiment and strategic insights.

4. **Privacy & Local Ownership:**
   Keep reasoning, core vector indexing, and memory processing local via Ollama and local embeddings.