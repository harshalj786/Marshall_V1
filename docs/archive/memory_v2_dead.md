# Marshall V2 — Memory & State Persistence

## Memory Architecture Blueprint

Marshall V2 utilizes a **Dual-Layer Memory System**:

```text
┌─────────────────────────────────────────────────────────────┐
│                      Agent Execution                        │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────────┐┌─────────────────────────────┐
│  Short-Term Working Memory   ││   Long-Term User Memory     │
│  (LangGraph AgentState)      ││   (V1 ChromaDB Compressed)  │
├──────────────────────────────┤├─────────────────────────────┤
│ • Rolling Message History    ││ • Personal Context Profiles │
│ • Active Tool Call Outputs   ││ • Entity Relationships      │
│ • State Checkpoints          ││ • Persisted across sessions │
└──────────────────────────────┘└─────────────────────────────┘