# edge_ai_framework — modular split

This package is your original `AbstractIntegratedModule_documented__3_.py`
(24,313 lines, ~44 classes) split into one file per concern.

## Layout

| File | Contains |
|---|---|
| `common.py` | All shared imports/setup, enums & small dataclasses (`Message`, `SecurityConfig`, `TrustLevel`, ...), `Singleton` |
| `weight_shaping.py` | `GeometricWeightShaping` |
| `activations.py` | `sigmoid`, `sigmoid_deriv`, `tanh_deriv`, `Activation`, `Loss` |
| `transformer.py` | `Transformer` |
| `layers.py` | `Dense`, `SoftmaxOutput` |
| `mlp.py` | `MLP` |
| `lstm.py` | `LSTMCell`, `LSTMNetwork`, `LSTMEngine` |
| `ensemble.py` | `WeightedEnsemblePredictor` |
| `automation.py` | `CrossSessionAutomation`, `AutoBatcherAutomation` |
| `explainability.py` | `ExplainabilityModule` |
| `model_storage.py` | `ModelStorage` |
| `messaging.py` | `AsyncMessageQueue`, `ThreadedMessageQueue` |
| `distributed_inference.py` | `AgentDistributedInference` |
| `query_node.py` | `QueryNode` |
| `pipeline.py` | `IntegratedPipeline` |
| `caching_security.py` | `AccurateAnswerCache`, `RateLimiter`, `InputSanitizer`, `ApiKeyManager` |
| `async_manager.py` | `AsyncResultQueue`, `WorkerPool`, `PipelineAsyncManager` |
| `prediction_manager.py` | `PipelinePredictionManager` |
| `peer_agent.py` | `ConsecutivePeerAgent` |
| `deployment.py` | `CohesiveAgentDeployment`, `run_secure_agent_cluster`, `example_async_with_result_queue` |
| `main.py` | The original demo/CLI entry point (`PermissiveTest`, `AsyncWrappertest`, etc.) |
| `__init__.py` | Re-exports everything so `from edge_ai_framework import X` works exactly like before |

Every class kept its original name, code, and behavior — nothing was rewritten,
only relocated. All 44 classes / top-level functions were accounted for and
verified: total line count matches the original file (the only additions are
per-file headers and import lines).

## How to use it

Drop the whole `edge_ai_framework/` folder next to where your script used to
live (so it sits alongside `AbstractOptimizedModules` / `abstract_weights_core`
if you use those optional accelerators), then:

```python
from edge_ai_framework import IntegratedPipeline, MLP, Transformer, WeightedEnsemblePredictor
```

This is a drop-in replacement for the old `from AbstractIntegratedModule import ...` line —
every public class is re-exported from the package root in `__init__.py`.

To run the original interactive demo:

```bash
python -m edge_ai_framework.main
```

## Why some files import each other "oddly"

The original code has real circular relationships — e.g. `MLP` and
`Transformer` reference each other, `IntegratedPipeline` and
`WeightedEnsemblePredictor` reference each other, and so on. A plain
`from .transformer import Transformer` at the top of `mlp.py` (while
`transformer.py` does `from .mlp import MLP` at its own top) would crash on
import with a circular-import error.

To keep every file honestly separated (rather than merging things back
together to dodge the cycles), sibling modules are imported as whole modules:

```python
from . import transformer
...
transformer.Transformer(...)
```

instead of:

```python
from .transformer import Transformer
...
Transformer(...)
```

This is functionally identical — it only changes how the reference is
spelled — and it's the standard fix for this kind of circular dependency.
None of the split classes inherit from a class in another file, and none of
these cross-referenced classes are used at class-definition time or as a
default argument, so this pattern is safe everywhere it's used.

## What was verified

- Every one of the original 24,313 lines was accounted for exactly once (no
  gaps, no duplication).
- All 21 resulting files parse as valid Python (`ast.parse`).
- The full package (`import edge_ai_framework`) imports successfully end to
  end with no circular-import errors.
- Spot-checked that key classes (`MLP`, `Transformer`, `IntegratedPipeline`,
  `WeightedEnsemblePredictor`) still expose their full original method sets
  after the split.

## What wasn't changed

This was a structural split only — no logic was modified, renamed, or
"cleaned up." If you'd also like the giant classes (e.g. `IntegratedPipeline`
at ~5,000 lines, `PipelinePredictionManager` at ~2,700 lines) broken down
internally, or the redundant per-file `try/except` optional-accelerator
imports consolidated into `common.py` only, that's a good next step — just ask.
