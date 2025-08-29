# Custom Strategies

Define your own multi‑stage orchestration logic in Python and run it with `--strategy` using a file or module spec.

## Specs: file or module

- File path: `path/to/strategy.py`
- File + class: `path/to/strategy.py:ClassName`
- Module + class: `package.module:ClassName`
- Module (single Strategy exported or `STRATEGY` provided): `package.module`

Security note: loading a strategy executes Python code. Only load trusted code.

## Base API

Subclass `Strategy` and implement `execute(self, prompt, base_branch, ctx) -> List[InstanceResult]`.

- Name: by default derived from the class name (e.g., `BestOfNStrategy` → `best-of-n`). You can also set `NAME = "my-name"`.
- Config: optionally subclass `StrategyConfig` and return it from `get_config_class()`; use `self.create_config()` to parse and validate values (CLI `-S key=value` and config file entries are merged in).
- Logging: use `self.logger` instead of importing logging.
- Return: a list of `InstanceResult` objects (often just one item).

Context methods

- `ctx.run(task_dict, key=...) -> Handle`: schedule a durable task; keys identify tasks and drive deterministic branch names.
- `await ctx.wait(handle) -> InstanceResult`: wait for one task; raises on failure (use try/except to tolerate).
- `await ctx.wait_all(handles, tolerate_failures=False) -> List[InstanceResult]` or `(successes, failures)` when tolerate_failures=True.
- `ctx.key(*parts) -> str`: build stable keys like `ctx.key("gen", i)`.
- `ctx.rand() -> float`: deterministic 0..1 for tie‑breakers and randomized choices.

Common task fields

- Minimal: `{ "prompt": str, "base_branch": str, "model": str }`
- Optional: `import_policy`, `import_conflict_policy`, `skip_empty_import`, `max_turns`, `network_egress`, `plugin_name`.

## Minimal example (single task)

Create a file `my_strategy.py` in your repo:

```python
from dataclasses import dataclass
from typing import List
from pitaya.orchestration.strategies.base import Strategy, StrategyConfig
from pitaya.shared import InstanceResult

@dataclass
class MyConfig(StrategyConfig):
    # Add your own knobs; model is inherited
    greeting: str = "Hello from Pitaya"

class MyStrategy(Strategy):
    NAME = "my-simple"

    def get_config_class(self) -> type[StrategyConfig]:
        return MyConfig

    async def execute(self, prompt: str, base_branch: str, ctx) -> List[InstanceResult]:
        cfg: MyConfig = self.create_config()  # type: ignore
        task = {
            "prompt": f"{prompt}\n\nAlso write HELLO.txt with: {cfg.greeting}",
            "base_branch": base_branch,
            "model": cfg.model,
        }
        h = await ctx.run(task, key=ctx.key("gen"))
        res = await ctx.wait(h)
        return [res]
```

Run it:

```bash
pitaya "Create a HELLO.txt file with 'Hello from Pitaya' text in it and commit it" \
  --strategy ./my_strategy.py -S greeting="Hello from Pitaya"
```

## Parallel fan‑out example

Spawn N parallel tasks and wait for all, tolerating failures:

```python
from dataclasses import dataclass
from typing import List
from pitaya.orchestration.strategies.base import Strategy, StrategyConfig
from pitaya.shared import InstanceResult

@dataclass
class FanOutConfig(StrategyConfig):
    n: int = 2

class FanOutStrategy(Strategy):
    def get_config_class(self) -> type[StrategyConfig]:
        return FanOutConfig

    async def execute(self, prompt: str, base_branch: str, ctx) -> List[InstanceResult]:
        cfg: FanOutConfig = self.create_config()  # type: ignore
        handles = []
        for i in range(int(cfg.n)):
            t = {"prompt": f"{prompt}\n\nVariant #{i+1}", "base_branch": base_branch, "model": cfg.model}
            h = await ctx.run(t, key=ctx.key("gen", i+1))
            handles.append(h)
        successes, failures = await ctx.wait_all(handles, tolerate_failures=True)
        # Optionally add selection logic, e.g., pick first success
        return successes or failures
```

Run it:

```bash
pitaya "Write the funniest and most original joke possible" \
  --strategy ./fanout.py -S n=3
```

## Loading from a module

If your strategy is installed/importable as a module:

```bash
pitaya "task" --strategy mypkg.my_mod:MyStrategy
# or (single Strategy in module)
pitaya "task" --strategy mypkg.my_mod
```

## Tips

- Keep prompts explicit about commit behavior when you expect a branch (e.g., “commit once with message …”).
- Use `ctx.key(...)` for every scheduled task to get deterministic branches and better resume behavior.
- Attach selection metadata to results by mutating `result.metrics`/`result.metadata` before returning, as shown in the built‑in strategies.
- Study the examples in `examples/` for patterns:
  - `examples/custom_simple.py`
  - `examples/fanout_two.py`
  - `examples/propose_refine.py`
