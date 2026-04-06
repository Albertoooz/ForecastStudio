# Foundation Prompt

You are working on an early-stage startup building an AI-powered forecasting platform
based on multi-agent orchestration, conversational interaction, and AutoML for time series.

## Core Principles (highest priority first)

1. **KISS** — keep everything as simple as possible.
2. **Explicit over implicit** — clear, structured interfaces.
3. **Readability over cleverness** — code should be obvious.
4. **Fewer abstractions, fewer layers, fewer files** — only add when necessary.
5. **Deterministic, debuggable behavior** — no hidden magic.

## What to Avoid

- Over-engineering.
- Premature abstractions.
- Complex frameworks when plain Python is sufficient.
- Deep inheritance trees.
- "Future-proofing" without concrete use cases.
- Autonomous AI behavior without human control.

## Architecture Rules

- Clear separation between:
  - product runtime agents (user-facing)
  - development-time agents (for code quality and evolution)
- Each module must have a single, clear responsibility.
- Each agent must:
  - have a narrow role
  - accept structured input
  - return structured output
  - never contain business logic hidden in prompts

## LLM / Agent Rules

- LLMs are planners and advisors, never executors.
- All LLM outputs must be structured (JSON or equivalent).
- No hidden decisions or side effects.
- Every decision must be inspectable and overridable by a human.
- If information is missing, ask explicitly instead of guessing.

## ML-Specific Rules

- Separate orchestration from modeling.
- Models must follow simple, explicit interfaces.
- Feature engineering must be declarative and reproducible.
- Prefer transparent models where possible.
- Avoid magic heuristics.

## Error Handling & Quality

- Fail fast with clear error messages.
- Validate inputs explicitly.
- Do not silently ignore errors.
- Prefer clarity over brevity when handling failures.

## Default Behavior

If a requirement is ambiguous:
- choose the simplest reasonable interpretation
- clearly state assumptions
- do not invent complexity

## Working Style

- Build incrementally.
- Start with minimal viable implementations.
- Improve only when a real need appears.
- Before adding anything new, ask:
  "Can this be done clearly without adding another abstraction?"
