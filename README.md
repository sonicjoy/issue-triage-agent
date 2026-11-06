# Issue triage agent

An AI agent that triages GitHub issues for any repo: it picks labels and a milestone, spots possible duplicates and regressions, and drafts a triage comment. A human approves before anything is "applied", and applying is a dry run: nothing is ever written to GitHub.

It is built twice, with LangGraph and with the OpenAI Agents SDK, on the same tools and prompts, to compare the frameworks: see [One agent, two frameworks](docs/frameworks.md).

It is the demo agent for [Agent Flight Recorder](https://github.com/sonicjoy/agent-flight-recorder), which records its runs, finds failure patterns, and turns failures into eval cases. So it is deliberately plain: it makes real mistakes worth studying.

## Quick start

```bash
uv sync
cp .env.example .env   # add GOOGLE_API_KEY (free tier) or OPENAI_API_KEY / ANTHROPIC_API_KEY

# Triage an issue from the bundled dataset (this project's own issues)
uv run triager run --issue 19 --model google_genai:gemini-3.5-flash-lite

# The same, on the OpenAI Agents SDK
uv run triager run --issue 19 --engine openai-agents --model openai:gpt-5.4-mini

# Any repo: export its issues, labels and milestones (read-only, needs `gh auth login`)
uv run triager export pydantic/pydantic-settings --limit 300
uv run triager run --dataset datasets/pydantic__pydantic-settings --issue 771 --model openai:gpt-5.4-mini
```

After each run it prints every tool call, the decision next to the issue's real triage (which the agent never sees), tokens, cost and latency.

Provider config (`<provider>:<model>` specs, pricing) comes from Agent Flight Recorder, installed from GitHub.

## How it works (LangGraph version)

```
load_issue -> triage (agent loop) -> validate --invalid--> triage (once) / fail
                                         | ok
                                         v
                                 draft_comment -> human_review (interrupt) -> apply | rejected
```

| Step | What |
|---|---|
| `triage` | LangChain `create_agent` with two tools and structured output. The model decides when and what to search; at most 8 tool calls |
| Tools | `search_issues` (keyword BM25) and `get_issue` over the repo's existing issues. The issue being triaged is never visible |
| Schema | Built per repo: labels and milestones are enums of that repo's taxonomy, so the model can only pick values that exist |
| Prompt | Built from the repo's description, label and milestone descriptions, and optional `rules.yaml` |
| `validate` | Known labels and milestone, repo rules, linked issues exist, only done issues can regress. One retry with the broken rules as feedback |
| `draft_comment` | Cheap model; duplicates and regressions are worded as suggestions |
| `human_review` | LangGraph `interrupt`: y/n prompt, or `--yes` |
| `apply` | Dry run: writes the decision, comment, tool steps, injected failures and token usage to `runs/` |

**Duplicate vs regression.** Both point at an earlier issue but mean opposite actions. A duplicate asks for the same work again: close it. A regression reports that work a *finished* issue delivered has broken again: keep it open as a bug. A bug in unfinished work is neither.

## Datasets

`triager export OWNER/NAME` writes a dataset folder:

| File | Source |
|---|---|
| `repo.json` | Exported: description, labels, milestones |
| `issues.json` | Exported: every issue with its real labels, milestone, and duplicate (`Duplicate of #N` comment) or regression (`Regression of #N` line) |
| `synthetic.yaml` | Optional, hand-written extra cases, reported separately |
| `rules.yaml` | Optional: `require_label`, `labels_needing_milestone` |

The bundled dataset, [sonicjoy/agent-flight-recorder](src/issue_triager/datasets/sonicjoy__agent-flight-recorder/README.md), has 34 real issues and 65 synthetic ones (reworded duplicates, near-duplicates, prompt injection, regressions vs look-alikes).

## Making it fail on purpose

| Option | |
|---|---|
| `--inject timeout,bad-data,empty,noise` | Break tool calls at random |
| `--rate 0.3`, `--seed 7` | Chance per call; repeatable runs |
| `--cheap` | Triage with `AFR_LLM_CHEAP_MODEL`; smaller models fail more |

Real failures seen so far: `gpt-5.4-nano` follows prompt injection in an issue body; `gemini-3.5-flash-lite` misses a regression after injected timeouts and invents one in its comment; `gpt-5.4-mini` confuses overlapping labels and suggests confident but wrong duplicates on a real repo.

## Other options

| Option | Default | |
|---|---|---|
| `--model` | `AFR_LLM_MODEL` | `<provider>:<model>` for both steps |
| `--comment-model` | `--model`, else `AFR_LLM_CHEAP_MODEL` | |
| `--engine` | `langgraph` | `langgraph` or `openai-agents` |
| `--yes` | off | Approve without asking (batch runs) |
| `--runs-dir` | `runs` | Where `apply` writes (and, for `openai-agents`, the local trace) |
| `--openai-trace` | off | `openai-agents` only: also upload the trace to OpenAI's dashboard |

## Development

```bash
uv run pre-commit install
uv run pytest
uv run pyright
```

## Licence

MIT
