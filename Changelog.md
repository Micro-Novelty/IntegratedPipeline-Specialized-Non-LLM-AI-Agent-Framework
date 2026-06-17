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

### Note: rest of documented versions from [0.1.5] -> [0.2.4] got deleted because of unintended overwrite in this repository to delete all files.

### [0.2.4] 2026-06-14
- fixed undefined variables bug in calibrate probs
- fixed nan fragility in trust scoring in P2P.

### [0.2.5] 2026-06-14
- Added new dynamic gate for Transformer Fixed and dynamic switching condition.
- Added Capabilities for the Model to save the Transformer weights as binaries in local SQlite database.

### [0.2.6] 2026-06-15
- Transformer is now Optimized using Cython!
- Refined memory checks to reduce further corruption in IntegratedPipeline memory.
- Added safety guards for cosine similarity to prevent float cascading through the flow.

### [0.2.7] 2026-06-16
- New cache management system for Transformers Cache
- Optimized Transformer Training time with less Computational overhead.
- Optimized Transformer attention quality computing with more newer Conditional gate for quality computing.

### [0.2.8] 2026-06-17
- Added safety guards for Cosine similarity to prevent string arrays to cause further errors.
- Refined csv load labels function to accept path dir, so users can finnaly find their Training labels without much restrictions and errors.
- Fixed bug in advanced prediction block that caused lstm weights to be set None.

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
