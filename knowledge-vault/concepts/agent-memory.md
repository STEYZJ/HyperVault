---
tags:
  - ai
  - memory
type: concept
priority: high
---

# Agent Memory Architecture

Long-term memory stores durable preferences, research state, project decisions, and
summaries that an agent should retrieve before ordinary notes.

## Retrieval Priority

Memory chunks receive a retrieval boost. Ordinary notes remain searchable, but memory
notes are ranked higher when the semantic and lexical evidence is comparable.

```python
def memory_boost(score: float) -> float:
    return score + 0.18
```

