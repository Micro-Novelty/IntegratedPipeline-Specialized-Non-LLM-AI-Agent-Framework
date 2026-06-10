## Changelog
All notable changes to this project will be documented in this file.

### [0.1.0] - 2026-05-05
[=] Added
Initial release of IntegratedPipeline
Memory-Augmented Neural Networks (MANN) support
Abstract Weight Encoder (AWE) implementation
Specialized MLP with noise robustness
Optimized Transformer with Alpha-based computing
Peer-to-Peer agent coordination
SQLite-based memory persistence
Multi-platform support (Windows, Linux, Raspberry Pi)

### [0.1.1] - 2026-05-08 
[=] Specific Bug fixed for P2P Robustness, in _handle_distributed_connections(), removing undefined variables that caused specific Failed connection.

### [0.1.2] - 2026-05-11
[=] Bug fixed:
- fix: conflicting roles in _handle_distributed-connections() in specific cases
- fix: fix check rate limit unbounded grow
- fix: added another authentication identification for P2P in handle client function
- fix connect_agent function contains undefined variables in last condition.

### [0.1.3] 2026-05-12
[=] Bug fixed:
- conflicting str and float during confidence handling in advanced prediction method

### [0.1.4] 2026-05-22
- More Robust advanced prediction than v 0.1.3
- Asynchronous Prediction capabilities, PipelineAsyncManager (Queue message handling, based architecture)
- New robust P2P Architecture
- CohesiveAgentDeployment (For asynchronous Prediction feature between peers, Hybrid based architecture that supports hybrid capabilities, synchronous prediction and asynchronous from ensemble prediction from peers)
- ConsecutivePeerAgent (Handles Asynchronous ensemble weighting from incoming peer, when CohesiveAgentDeployment is busy, or cant capture the prediction request message)

### [0.1.5] 2026-05-24 (Officially tested and validated in ARM64 Env.)
- More Robust asynchronous advanced prediction than v 0.1.4
- Updated P2P Architecture
- Ensuring singleton on IntegratedPipeline
- fixed fragility in feature parsing from json file
- Officially tested in ARM64 Environment using docker

### [0.1.6] 2026-05-31 
- More robust meta ensemble method
- fixed bugs in advanced_prediction_method that happens without Transformer

### [0.1.7] 2026-06-01
- More Robust Transformer
- Transformer has new features that helps and coordinate with alpha.

### [0.1.8] 2026-06-02
- batching features for transformer

### [0.2.0] 2026-06-04
- P2P Finnaly Proven works in ARM64 Docker environment with QEMU.

### [0.2.1] 2026-06-07
- new LSTM Architecture added
- Ensemble weighting updated with LSTM
- fixed undefined variables in CohesiveAgentDeployment.

### [0.2.2] 2026-06-09
- fixed bug in Geometric weight shaping
- fixed bug in LSTM Weight generation.

### [0.2.3] 2026-06-10
- fixed bug in start server handling timeout
- fixed bug in server shutdown that caused code block execution

### [=] Features
Continuous learning without catastrophic forgetting
Local AI orchestrator with SQLite database
Hybrid MLP + Transformer ensemble predictions
Distributed agent system with SSL/TLS security
Explainability module for predictions
Cross-session model automation
Asynchronous local prediction (Worker based Queue) and P2P Asynchronous feature (P2P using socket)

### [=] Supported Platforms
Windows x86_64
Linux x86_64
Raspberry Pi (ARM64)

### [=] Known Limitations
Raspberry Pi: installation takes 30+ minutes
Large dataset optimization (1M+ samples) pending
