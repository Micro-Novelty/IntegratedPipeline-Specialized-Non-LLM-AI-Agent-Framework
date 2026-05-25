# ---------------------------------------------------------------------------
# aim/__init__.py
# Public API re-exports for the AbstractIntegratedModule package.
#
# Dependency order (bottom → top):
#   primitives → security → messaging
#   primitives → geometry → nn → mlp → transformer
#   mlp + transformer + geometry → ensemble
#   mlp + transformer + geometry + storage → explainability
#   messaging + geometry → inference
#   all of the above → pipeline
#   security + messaging + mlp + transformer → async_manager
#   pipeline + async_manager + messaging + security → agents
# ---------------------------------------------------------------------------

from .primitives import (
    MessagePriority, WrapperState, AsyncTask,
    TrustLevel, RequestStatus, AsyncRequest,
    SecureMessage, SingletonMeta, Singleton,
)
from .security import (
    SecurityConfig, SecurityLevel, SecurityError, AdminRole,
    RateLimiter, InputSanitizer, ApiKeyManager,
)
from .messaging import (
    Message, AsyncMessageQueue, ThreadedMessageQueue,
    AsyncResultQueue, WorkerPool,
)
from .geometry import GeometricWeightShaping
from .nn import Activation, Loss, Dense, SoftmaxOutput
from .mlp import MLP
from .transformer import Transformer
from .storage import ModelStorage, CrossSessionAutomation
from .explainability import ExplainabilityModule
from .ensemble import WeightedEnsemblePredictor
from .inference import QueryNode, AutoBatcherAutomation, AgentDistributedInference
from .pipeline import IntegratedPipeline
from .async_manager import PipelineAsyncManager, PipelinePredictionManager
from .agents import ConsecutivePeerAgent, CohesiveAgentDeployment

__all__ = [
    # Primitives
    "MessagePriority", "WrapperState", "AsyncTask",
    "TrustLevel", "RequestStatus", "AsyncRequest",
    "SecureMessage", "SingletonMeta", "Singleton",
    # Security
    "SecurityConfig", "SecurityLevel", "SecurityError", "AdminRole",
    "RateLimiter", "InputSanitizer", "ApiKeyManager",
    # Messaging
    "Message", "AsyncMessageQueue", "ThreadedMessageQueue",
    "AsyncResultQueue", "WorkerPool",
    # Core ML
    "GeometricWeightShaping",
    "Activation", "Loss", "Dense", "SoftmaxOutput",
    "MLP", "Transformer",
    # Pipeline components
    "ModelStorage", "CrossSessionAutomation",
    "ExplainabilityModule",
    "WeightedEnsemblePredictor",
    "QueryNode", "AutoBatcherAutomation", "AgentDistributedInference",
    "IntegratedPipeline",
    # Deployment
    "PipelineAsyncManager", "PipelinePredictionManager",
    "ConsecutivePeerAgent", "CohesiveAgentDeployment",
]
