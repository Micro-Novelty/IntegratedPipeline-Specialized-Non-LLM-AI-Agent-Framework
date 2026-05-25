# AbstractIntegratedModule (`aim`) — Package Structure

Refactored from a single monolithic file into focused, independently importable modules.

## Module Map

```
aim/
├── __init__.py          # Public re-exports for the whole package
│
├── primitives.py        # Enums, lightweight dataclasses, Singleton base
├── security.py          # SecurityConfig, rate limiter, input sanitizer, API key manager
├── messaging.py         # Async/threaded queues, request/result tracking
│
├── geometry.py          # GeometricWeightShaping (GWS) — data-adaptive weight init
├── nn.py                # Activation, Loss, Dense, SoftmaxOutput
├── mlp.py               # MLP — multi-layer perceptron with focused sub-network
├── transformer.py       # Transformer — single-layer with manual backprop
│
├── storage.py           # ModelStorage (SQLite), CrossSessionAutomation
├── explainability.py    # ExplainabilityModule — attributions, feedback loop
├── ensemble.py          # WeightedEnsemblePredictor — dynamic/meta ensemble
│
├── inference.py         # QueryNode, AutoBatcherAutomation, AgentDistributedInference
├── pipeline.py          # IntegratedPipeline — top-level orchestrator
│
├── async_manager.py     # PipelineAsyncManager (HTTP), PipelinePredictionManager
└── agents.py            # ConsecutivePeerAgent, CohesiveAgentDeployment
```

## Dependency Graph

```
primitives  ──────────────────────────────────────────────────┐
security    (← primitives)                                    │
messaging   (← primitives)                                    │
                                                              │
geometry    ──────────────────────────┐                       │
nn          (← geometry)             │                       │
mlp         (← geometry, nn)         │                       │
transformer (← geometry, nn)         │                       │
                                     │                       │
storage     (← geometry)             │                       │
explainability (← geometry,nn,mlp,transformer)               │
ensemble    (← geometry,mlp,transformer,storage)             │
                                     │                       │
inference   (← primitives,messaging,geometry,               │
               mlp,transformer,ensemble,storage)             │
                                     │                       │
pipeline    (← geometry,nn,mlp,transformer,                  │
               ensemble,storage,inference)                   │
                                     │                       │
async_manager (← security,messaging,primitives,mlp,transformer) │
agents      (← pipeline,async_manager,messaging,security)   ┘
```

## Usage

```python
# Import the whole package
import aim

pipeline = aim.IntegratedPipeline(...)
pipeline.train(X, y)
pred = pipeline.predict(["some text"])

# Or import only what you need
from aim.geometry import GeometricWeightShaping
from aim.mlp import MLP
from aim.transformer import Transformer
from aim.pipeline import IntegratedPipeline
from aim.agents import CohesiveAgentDeployment
```

## Circular Import Notes

Two pairs of modules have a mutual dependency that is broken with lazy imports:

| Pair | Direction | Resolution |
|---|---|---|
| `ensemble` ↔ `inference` | `WeightedEnsemblePredictor` uses `QueryNode` | `QueryNode` imported inside the method body |
| `inference` ↔ `pipeline` | `AutoBatcherAutomation` uses `IntegratedPipeline` | `IntegratedPipeline` imported inside the method body |
| `pipeline` ↔ `async_manager` | `PipelineAsyncManager` wraps `IntegratedPipeline` | Pipeline injected via `__init__` argument (no import needed) |
