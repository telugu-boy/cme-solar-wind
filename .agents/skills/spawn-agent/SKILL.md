---
name: spawn-agent
description: Spawns an isolated background subagent running Gemini 3.6 Flash to execute code implementation, run tests, or handle file updates.
---

# Spawn Agent Skill

When instructed to spawn a background worker or delegate an implementation task:
1. Target **Gemini 3.6 Flash** for the subagent thread to minimize context overhead on the primary 3.1 Pro orchestrator.
2. Pass an explicit, narrow set of file paths and clear instructions (avoid sending whole repository dumps).
3. Instruct the subagent to report back strictly with artifact diffs, test pass/fail results, or execution logs.