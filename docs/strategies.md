# Strategies

Pitaya ships with several production‑ready strategies. Select one with `--strategy` and pass parameters using `-S key=value`.

Notes

- `-S` auto‑parses numbers and booleans; strings should be quoted. For lists or complex values, prefer a `pitaya.yaml` file under the `strategies:` section.
- The `model` is chosen via `--model` (applies to the selected strategy); you can also override it per strategy in `pitaya.yaml`.

## simple

One agent, one branch. No extra phases.

- Options: inherits only `model`.
- Example:

```bash
pitaya "Create a HELLO.txt file with 'Hello from Pitaya' text in it and commit it"
```

## scoring

Generate a solution, then evaluate it with a separate read‑only scoring task that returns JSON with a numeric `score` and rationale. Score and feedback are attached to the generation result.

Key options

- `scorer_model`: scoring model (defaults to generator model when empty)
- `weight_correctness`, `weight_completeness`, `weight_quality`: non‑negative weights
- `score_scale_max`: 10 or 100
- `fail_below_score`: if set, mark result failed when below threshold
- `scorer_max_retries`: additional attempts to obtain valid JSON
- `read_only_scoring`: if true, scoring uses `import_policy=never`
- `max_turns`: optional runner hint for the scoring task

Examples

```bash
# Evaluate implementation quality, require score >= 7/10
pitaya "implement robust CSV import" \
  --strategy scoring -S score_scale_max=10 -S fail_below_score=7

# Use a separate scoring model
pitaya "refactor config loader" \
  --strategy scoring -S scorer_model=opus
```

Result metadata/metrics (subset)

- `metrics.score` (normalized to the chosen scale), `metrics.feedback`, `metrics.score_scale_max`
- `metadata.score`, `metadata.feedback`, `metadata.scorer_model`, `metadata.scorer_success`

## best-of-n

Run N candidates in parallel, score each (via `scoring` internally), and select the best.

Key options

- `n`: number of candidates (1–50)
- `tie_breaker`: `first` or `random` (deterministic)
- `require_min_success`: minimum number of successful candidates required
- Scoring passthrough: `scorer_model`, weights, `score_scale_max`, `fail_below_score`, `scorer_max_retries`, `read_only_scoring`, `max_turns`

Examples

```bash
# 5 candidates, select highest score
pitaya "Write the funniest and most original joke possible" \
  --strategy best-of-n -S n=5

# Require at least 2 successful candidates; break ties randomly
pitaya "optimize build pipeline" \
  --strategy best-of-n -S n=5 -S require_min_success=2 -S tie_breaker=random
```

Selection metadata (subset)

- `metrics.selected=true`, `metrics.selection_reason`
- `metrics.bestofn_index`, `metrics.bestofn_total`

## iterative

Refine a solution across fixed rounds: generate → review (read‑only) → refine on the same branch. Stops early when configured and no changes are made.

Key options

- `iterations`: number of rounds (1–10)
- `reviewer_model`: model for review tasks (defaults to generator model)
- `review_max_retries`: additional attempts to obtain good feedback
- `read_only_review`: if true, reviewers use `import_policy=never`
- `max_turns`: optional runner hint for review tasks
- `stop_on_no_changes`: stop when an iteration yields no changes

Example

```bash
pitaya "Write the funniest and most original joke possible" \
  --strategy iterative -S iterations=3
```

## bug-finding

Two phases: discovery (read‑only) then validation. Discovery produces a structured bug report in the final message. Validation independently reproduces the bug from the base branch, writes `BUG_REPORT.md`, and commits exactly once if valid.

Key options

- `target_area` (required): area to explore (e.g., `src/parser`)
- `bug_focus`: extra guidance (e.g., "race conditions in cache")
- `report_path`: default `BUG_REPORT.md`
- `discovery_max_retries`, `validation_max_retries`
- `read_only_discovery`: prevent commits during discovery
- `validator_model`: optional model override for validation

Example

```bash
pitaya "find a real bug we can reproduce" \
  --strategy bug-finding -S target_area=src/parser
```

Result notes

- Validation success requires one commit that includes `report_path` in the branch. Metadata includes `bug_confirmed` and (when confirmed) `bug_report_branch`.

## doc-review

Multi‑stage technical documentation review:

1) Reviewers (N per page) produce raw reports under `reports/doc-review/raw/REPORT_{slug}__r{n}.md`.
2) Validators refine and validate each reviewer report on its branch. The composer stage is intentionally removed; successful validator results are the output.

Key options

- `pages_file` (required): YAML/JSON list of `{title, path, slug?}`
- `reviewers_per_page`: default 1
- `report_dir`: default `reports/doc-review`
- `reviewer_max_retries`, `validator_max_retries`

Example pages.yml

```yaml
- title: Getting Started
  path: docs/getting-started.md
- title: API Reference
  path: docs/api.md
```

Run the strategy

```bash
pitaya "Review docs" --strategy doc-review -S pages_file=pages.yml -S reviewers_per_page=2
```

---

Tips

- Prefer `-S` for simple numeric/boolean settings; put more complex lists (e.g., diversity hints) into `pitaya.yaml` under `strategies:`.
- Strategies create branches only when there are changes to import (default policy). You can adjust import behavior in configuration if needed.

## pr-review

Lightweight PR review orchestration for CI: run N reviewers, validate each, and compose a final summary suitable for posting to a pull request.

Flow

- Reviewers (N total): each inspects changes with local git in the isolated workspace (e.g., `git diff <compare_branch>...<workspace_branch>`) and returns a Markdown report.
- Validators (1 per reviewer): refine, dedupe, correct severities, and append a fenced JSON trailer with verdict/counts.
- Composer: aggregates validated reviewer messages and returns a consolidated review; sets an overall verdict (PASS or NEEDS_CHANGES).

Key options

- `reviewers` (default 3)
- `reviewer_max_retries` (default 0), `validator_max_retries` (default 1)
- `base_branch` (optional) — override the branch used as the diff baseline when different from the runner’s checkout branch.
- `include_branches` (list of unqualified branch names to make available read‑only in the workspace; use `pitaya.yaml`)
- `fail_on` severities that should fail CI (default `HIGH`)
- `ci_fail_policy` (default `needs_changes`; options: `needs_changes`, `always`, `never`)
- Long-form instructions: `review_instructions{,_path}`, `validator_instructions{,_path}`, `composer_instructions{,_path}`

Branch availability in the workspace

- The workspace is checked out to whatever branch you pass to Pitaya’s `--base-branch` flag (for PRs we recommend `pr_head`).
- The strategy compares against `base_branch` when provided; otherwise it picks the first branch listed in `include_branches` that differs from the checkout branch. If no such branch exists it falls back to the checkout branch.
- Additional read‑only branches can be included via `include_branches` (must exist as `origin/<name>`). No remotes are available and push is disabled.
- Pitaya also automatically includes the host repository’s current HEAD branch when it differs from the checkout branch.

Examples

```bash
# Basic (default comparison logic)
pitaya "Review this PR" --strategy pr-review -S reviewers=3

# Provide long custom guidance from files
pitaya "Review this PR" --strategy pr-review \
  -S review_instructions_path=.ci/reviewer.md \
  -S validator_instructions_path=.ci/validator.md \
  -S composer_instructions_path=.ci/composer.md

# Review a PR with the head checked out locally and diff against main
pitaya "Review this PR" --strategy pr-review \
  --base-branch pr_head \
  -S include_branches=main

# Include additional branches via -S (CSV)
pitaya "Review this PR" --strategy pr-review \
  --base-branch feature/new-ui \
  -S include_branches=main,release/1.2.0

# Or as a JSON list
pitaya "Review this PR" --strategy pr-review \
  --base-branch feature/new-ui \
  -S include_branches='["main","release/1.2.0"]'
```

pitaya.yaml example to include extra branches

```yaml
strategies:
  pr-review:
    reviewers: 3
    include_branches:
      - main        # primary comparison branch
      - release/1.2.0
```

CI integration

- Run Pitaya headless in your PR workflow and post the composer’s final_message as the review body. Verdict and counts are also present under `metrics.pr_review_verdict` / `metrics.pr_review_counts` in the composer instance JSON (results/<run_id>/instances/*.json).
