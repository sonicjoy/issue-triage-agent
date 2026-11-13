# Dataset: sonicjoy/agent-flight-recorder

The bundled dataset: Agent Flight Recorder's own issues, so the agent triages the project that records it.

| File | What | Edit by |
|---|---|---|
| `repo.json`, `issues.json` | The repo's labels, milestones and 35 issues with their real triage | `uv run triager export sonicjoy/agent-flight-recorder --out src/issue_triager/datasets/sonicjoy__agent-flight-recorder` |
| `synthetic.yaml` | 65 hand-written cases: 30 rewordings of real issues (duplicates of them) and 35 tricky ones (near-duplicates, ambiguous labels, cross-cutting, injection, out of scope, regressions vs look-alikes) | Hand |
| `rules.yaml` | Every issue needs a label; `stretch` needs a milestone | Hand |
| `accepted_answers.yaml` | Answers accepted besides real issues' exported truth, each with a reason (six, from AFR #14's error analysis). The truth, and what the agent's tools show, stay as exported. Synthetic duplicates of an issue accept the same, and graded runs record the reason as `triager.accept_reason` | Hand |

Load with `issue_triager.dataset.load_dataset()`. Each case has a `truth` the agent is graded against; give the agent only `case.agent_input()` (id, title, body), never the target's labels or milestone. Report scores separately for `source: real` and `source: synthetic`.

**Accepted answers:** a case's `truth.accept` lists other answers that also count as right, per field: in `synthetic.yaml` (e.g. s58's `accept: {duplicate_of: [null]}`, a debatable duplicate) or, for real issues, in `accepted_answers.yaml`. Graded runs list them as `triager.accepted.<field>`.

**Caveat:** the real issues are planned tasks written and labelled in one sitting, so they are tidier than a typical tracker and their labels fit unusually well. Try an exported public repo for a harder test.

Duplicate vs regression (both link an earlier issue, never both at once):

- **Duplicate** asks for the same work again, done or not. Close it and link the original. Marked by a comment that starts `Duplicate of #N`, GitHub's own convention.
- **Regression** reports that work a closed, completed issue delivered has broken again. Keep it open as a bug. Marked by a line `Regression of #N` in the body or a comment.
- A bug in behaviour that never worked, or in unfinished work, is neither.
