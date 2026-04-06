# Agent Prompts

This directory contains prompt templates for different agent roles in the development workflow.

## Philosophy

Following the "Atlas" approach from Reddit:
- **Multiple roles**, not one AI
- **Multiple prompts**, not one prompt
- **Manual orchestration**, not automation
- **Full control**, not magic

## Available Agents

### 🏗️ Architect Agent (`architect.md`)
Reviews architecture and design. Protects simplicity and maintainability.

**Use when:**
- After implementing a feature
- Before major refactoring
- When architecture feels complex

**Output:** Structural review, simplification recommendations

---

### 🧹 Code Quality Agent (`code_quality.md`)
Reviews code style, naming, readability. Local improvements only.

**Use when:**
- After Architect Agent approves structure
- Before committing code
- When code feels unclear

**Output:** Naming issues, complexity suggestions, style fixes

---

### 🧪 Test Agent (`test_agent.md`)
Identifies missing tests and edge cases. Ensures correctness.

**Use when:**
- After code is written and reviewed
- Before merging to main
- When adding new features

**Output:** Test suggestions, edge cases, determinism checks

---

### 📦 Product Agent (`product_agent.md`)
Reviews user workflows and runtime behavior. Ensures explicit control.

**Use when:**
- Designing new features
- Reviewing agent interactions
- Before production deployment

**Output:** Workflow review, control point analysis

---

### 🔬 Research Agent (`research_agent.md`) - Optional
Researches technical options and makes recommendations.

**Use when:**
- Before adding new dependencies
- Choosing between algorithms
- Evaluating libraries

**Output:** Comparison of options, specific recommendation

---

## How to Use

**📖 See [CURSOR_GUIDE.md](./CURSOR_GUIDE.md) for detailed step-by-step instructions on using agents in Cursor.**

### Quick Start

1. Open the relevant prompt file (e.g., `architect.md`)
2. Copy the entire content (Cmd+A, Cmd+C)
3. Open Cursor Chat (Cmd+L)
4. Paste the prompt
5. Add your specific task:
   ```
   Review the current repository structure
   ```
   or
   ```
   Review forecaster/interface/conversation.py for architectural issues
   ```

## Workflow Example

Typical development cycle:

1. **Write code / feature**
2. **Run Architect Agent**: Review structure
   ```
   [Paste architect.md prompt]
   Review the new feature in forecaster/models/
   ```
3. **Fix architectural issues**
4. **Run Code Quality Agent**: Review style
   ```
   [Paste code_quality.md prompt]
   Review forecaster/models/new_model.py for code quality
   ```
5. **Fix style issues**
6. **Run Test Agent**: Identify tests needed
   ```
   [Paste test_agent.md prompt]
   What tests are missing for the new forecasting model?
   ```
7. **Write tests**
8. **Merge**

## Principles

- **Manual orchestration**: You decide when to run which agent
- **No automation**: Agents don't call each other
- **Full control**: Every decision is yours
- **Deterministic**: Same input → same output
- **Auditable**: All agent outputs are visible

## Foundation Prompt

All agents inherit from `foundation.md` which contains the core project principles. You can reference it or include it with agent prompts for context.

## When to Automate

**Not now.** Only automate when:
- Architecture is stable
- You have 2-3 complete workflows
- Tests are in place
- Manual process is proven

Then you can consider:
- Sequential agent runs
- Parallel reviews
- Automated checks in CI

But start with manual control.
