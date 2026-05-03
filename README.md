# IntegratedPipeline---Custom-AI-Agent-Core-library

[~] IntegratedPipeline is a standalone Custom AI Agent Library for memory Augmented Agentic Framework, Specifically designed to provide Agentic capability for any Autonomous Agentic Framework locally and Coordinatively that runs efficiently on High-end embedded systems, where the AI Can directly and continously learn datas with minimalcompute, with augmented memory init, Peer-To-Peer Sharing, And Explainability capability based on proof from in it's internal metrics, reducing Black-Box condition necessary for reliability.

<img width="393" height="385" alt="1000077388-removebg-preview" src="https://github.com/user-attachments/assets/c7794da0-f9c5-4c61-8b63-642700b965f5" />


# Introduction:
[=] Memory augmented Neural network (MANN) is a neural network architecture coupled with an external, dynamic memory module, allowing it to store, retrieve, and update information similarly to a computer's RAM. Unlike traditional networks that store knowledge only in weight parameters, MANNs excel at fast learning, long-term dependency handling, and episodic recall, In IntegratedPipeline, Its memory is stored in a custom database inside your local machine, then later used for memory retrieval, transfered to the AI Dictionary where it can finnaly recall its memory when input condition matched with memory.


# Main Components:
[=] With a total of 17 different stacked Architectures, The main Component's of IntegratedPipeline is:

1. 1. GeometricWeightShaping
      
Purpose: Analyzes the geometric structure of data (anisotropy, spectral properties, complexity) and generates optimal weight matrices based on that geometry. Essentially teaches the model how to "understand" the shape of data before processing it, Highly robust to noise, making it an excellent fit for messy environment.
---
2. Activation
   
Purpose: Provides standard neural network activation functions (ReLU, sigmoid, softmax) and their derivatives for backpropagation for MLP Class.
---
3. Loss
   
Purpose: Implements categorical crossentropy loss and its gradient for training classification models for both MLP and The Transformer
---
4. Transformer
   
Purpose: A complete transformer implementation with multi-head attention, positional embeddings, feed-forward networks, layer normalization, and custom backpropagation. Includes both fixed (stable) and dynamic (adaptive) training modes necessary for Scarce data environment using algorithm such as Alpha based computing directly during forward pass.
---
5. Dense
   
Purpose: A geometric-aware dense layer that adapts its weights based on input data geometry and handles variable input dimensions automatically.
---
6. SoftmaxOutput
   
Purpose: A simple wrapper around softmax activation that stores the output and passes gradients through unchanged (since softmax + crossentropy gradient is handled elsewhere).
---
7. MLP
   
Purpose: A multi-layer perceptron that can switch between standard training and "focused" training (using feed-forward layers only) based on data complexity. Includes prediction, scoring, and geometric measurement methods.
---
8. WeightedEnsemblePredictor
   
Purpose: Combines Transformer and MLP predictions using dynamic weighting based on attention quality, model confidence, and agreement. Also manages memory storage, explainability, and peer agent communication.
---
9. CrossSessionAutomation
    
Purpose: Manages exporting, importing, and syncing model sessions across different devices or time periods. Allows saving entire model states to JSON and transferring them over network sockets.
---
10. ExplainabilityModule
    
Purpose: Generates human-readable explanations for predictions, learns from user feedback, maintains decision history, and batch-trains on corrections. The transparency layer for the AI agent.
---
11. ModelStorage
    
Purpose: SQLite-based persistence for models, attention weights, node memories, and agent data. Handles serialization/deserialization of numpy arrays and model dictionaries.
---
12. AgentDistributedInference
    
Purpose: The distributed agent system - can act as a server or client, handles SSL/TLS security, rate limiting, authentication, peer-to-peer prediction requests, memory synchronization, ensemble voting, and trust management between agents.
--
14. QueryNode
    
Purpose: Manages trust relationships and identity verification between nodes. Evaluates node agreement, establishes connections, performs safety checks, and maintains the network of trusted peers.
---
14. AutoBatcherAutomation
    
Purpose: Automatically batches incoming prediction requests to optimize throughput. Collects requests up to a maximum batch size or time window, then processes them together.
---
15. IntegratedPipeline

Purpose: The main orchestration class that ties everything together - handles text encoding, model initialization, training, prediction, memory management, hybrid predictions, and distributed inference coordination.
---
16. PipelinePredictionManager

Purpose: High-level prediction interface that loads labels from CSV, performs regular/advanced/hybrid predictions, displays results, and calculates entropy for uncertainty estimation.


# Why IntegratedPipeline?
[~] IntegratedPipeline is a great choice for Edge-device AI Agentic framework and High-end Embedded systems, With its Custom Research-Grade Multi-Layer-Perceptron (AWE) and Enhanced Transformer Embedding that can directly tolerate scarce Data using Weighted Confidence assembling for better reliability over Messy environments, such as:

[1.] User data's : User data is often messy and ambiguous, The Tiny MLP will do the job for shaping the necessary Weight to complement for the Ambiguous noisy pattern, AWE MLP is highly robust to noise, proven in synthetic Environment such as scikit-learn Make-Classification scarce and Noisy Input robustness during generalization test. making it a great fit for messy, Scarce data.
[2.] Small Dataset's : We often don't have enough Dataset to train a Transformer Model, Thats Why IntegratedPipeline Offers a Highly optimized Transformer that supports scarce dataset processing, Using Alpha-Based computing as a Warm-up for training, it provides a direct Boost for the transformer to be efficient in scarce-data Environment. 
[3.] Non-Representative data (Undersampled) : IntegratedPipeline Support's Large ambiguous data that come's from file with format such as CSV Format to extract title's and label's necessary to create automatic Dataset for Later Training from the given dat'as, making it optimized for specific task's and easier dataset creation with lower overfitting rate for reliability.

# Requirements:
[~] To Quickly Run IntegratedPipeline, Requirements for it include's:
- Python 3.13+
- AbstractIntegratedModule.pyd (Main Module)
- AbstractIntegratedModule.dll (Main Module for Windows)
- Simple datasets for Input.


