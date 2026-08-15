# separated_module — modular split

This package is the original `AbstractIntegratedModule.py`
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
| `__init__.py` | Re-exports everything so `from separated_module import X` works |

