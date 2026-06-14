"""
AbstractIntegratedPipeline — module index
==========================================
Split from the monolithic source for easier architecture review.

File → Primary classes contained
---------------------------------
enums_and_dataclasses.py     MessagePriority, WrapperState, AsyncTask, TrustLevel,
                              RequestStatus, AsyncRequest, SecureMessage, Message,
                              SecurityConfig, SecurityLevel, SecurityError, AdminRole,
                              SingletonMeta, Singleton
lstm.py                       LSTMCell, LSTMNetwork, LSTMEngine
core_ml.py                    GeometricWeightShaping, Activation, Loss
transformer.py                Transformer
mlp.py                        Dense, SoftmaxOutput, MLP
ensemble.py                   WeightedEnsemblePredictor
cross_session.py              CrossSessionAutomation
explainability.py             ExplainabilityModule
model_storage.py              ModelStorage
messaging.py                  Message (agent version), AsyncMessageQueue,
                              ThreadedMessageQueue
agent_distributed_inference.py  AgentDistributedInference
query_node.py                 QueryNode
auto_batcher.py               AutoBatcherAutomation
integrated_pipeline.py        IntegratedPipeline
security_utils.py             RateLimiter, InputSanitizer, ApiKeyManager
async_worker.py               AsyncResultQueue, WorkerPool
pipeline_async_manager.py     PipelineAsyncManager
pipeline_prediction_manager.py  PipelinePredictionManager
consecutive_peer_agent.py     ConsecutivePeerAgent
cohesive_agent_deployment.py  CohesiveAgentDeployment
examples_and_tests.py         Test helpers, PermissiveTest, __main__ entry point
"""
