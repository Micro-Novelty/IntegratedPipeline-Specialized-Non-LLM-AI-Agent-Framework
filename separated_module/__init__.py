"""
edge_ai_framework
==================

This package is a modular split of the original monolithic
`AbstractIntegratedModule` file. Each concern now lives in its own file:

    common.py                - shared enums, dataclasses, Singleton, SecurityConfig
    weight_shaping.py        - GeometricWeightShaping
    activations.py           - sigmoid / tanh helpers, Activation, Loss
    transformer.py           - Transformer
    layers.py                - Dense, SoftmaxOutput
    mlp.py                   - MLP
    lstm.py                  - LSTMCell, LSTMNetwork, LSTMEngine
    ensemble.py               - WeightedEnsemblePredictor
    automation.py             - CrossSessionAutomation, AutoBatcherAutomation
    explainability.py         - ExplainabilityModule
    model_storage.py          - ModelStorage
    messaging.py               - AsyncMessageQueue, ThreadedMessageQueue
    distributed_inference.py   - AgentDistributedInference
    query_node.py              - QueryNode
    pipeline.py                 - IntegratedPipeline
    caching_security.py         - AccurateAnswerCache, RateLimiter, InputSanitizer, ApiKeyManager
    async_manager.py            - AsyncResultQueue, WorkerPool, PipelineAsyncManager
    prediction_manager.py       - PipelinePredictionManager
    peer_agent.py                - ConsecutivePeerAgent
    deployment.py                - CohesiveAgentDeployment + example runners
    main.py                      - original example/demo entry point (PermissiveTest)

Everything is re-exported here so existing code that used to do

    from AbstractIntegratedModule import IntegratedPipeline, MLP, ...

can switch to

    from edge_ai_framework import IntegratedPipeline, MLP, ...

with no other changes.
"""

from .common import (
    MessagePriority, WrapperState, AsyncTask, TrustLevel, RequestStatus,
    AsyncRequest, SecureMessage, Message, SecurityConfig, SecurityLevel,
    SecurityError, AdminRole, SingletonMeta, Singleton,
    logger, _OPT_AVAILABLE, _RUST_MODULE_AVAILABLE,
)

from .weight_shaping import GeometricWeightShaping
from .activations import sigmoid, sigmoid_deriv, tanh_deriv, Activation, Loss
from .transformer import Transformer
from .layers import Dense, SoftmaxOutput
from .mlp import MLP
from .lstm import LSTMCell, LSTMNetwork, LSTMEngine
from .ensemble import WeightedEnsemblePredictor
from .automation import CrossSessionAutomation, AutoBatcherAutomation
from .explainability import ExplainabilityModule
from .model_storage import ModelStorage
from .messaging import AsyncMessageQueue, ThreadedMessageQueue
from .distributed_inference import AgentDistributedInference
from .query_node import QueryNode
from .pipeline import IntegratedPipeline
from .caching_security import AccurateAnswerCache, RateLimiter, InputSanitizer, ApiKeyManager
from .async_manager import AsyncResultQueue, WorkerPool, PipelineAsyncManager
from .prediction_manager import PipelinePredictionManager
from .peer_agent import ConsecutivePeerAgent
from .deployment import CohesiveAgentDeployment, run_secure_agent_cluster, example_async_with_result_queue

__all__ = [
    "MessagePriority", "WrapperState", "AsyncTask", "TrustLevel", "RequestStatus",
    "AsyncRequest", "SecureMessage", "Message", "SecurityConfig", "SecurityLevel",
    "SecurityError", "AdminRole", "SingletonMeta", "Singleton",
    "GeometricWeightShaping",
    "sigmoid", "sigmoid_deriv", "tanh_deriv", "Activation", "Loss",
    "Transformer", "Dense", "SoftmaxOutput", "MLP",
    "LSTMCell", "LSTMNetwork", "LSTMEngine",
    "WeightedEnsemblePredictor",
    "CrossSessionAutomation", "AutoBatcherAutomation",
    "ExplainabilityModule", "ModelStorage",
    "AsyncMessageQueue", "ThreadedMessageQueue",
    "AgentDistributedInference", "QueryNode", "IntegratedPipeline",
    "AccurateAnswerCache", "RateLimiter", "InputSanitizer", "ApiKeyManager",
    "AsyncResultQueue", "WorkerPool", "PipelineAsyncManager",
    "PipelinePredictionManager", "ConsecutivePeerAgent", "CohesiveAgentDeployment",
    "run_secure_agent_cluster", "example_async_with_result_queue",
]
