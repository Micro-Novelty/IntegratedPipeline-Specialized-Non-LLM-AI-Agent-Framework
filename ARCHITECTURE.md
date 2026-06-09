# Architecture Documentation

## System Overview

IntegratedPipeline consists of 11 main interconnected components working together.

## High-Level Integrated Architecture
### IntegratedPipeline
### WeightedEnsemblePredictor (Ensemble weighting)
### AgentDistributedInference (P2P handling)
### Model storage (Database handling)
### ConsecutivePeerAgent (Ensemble P2P Handling)
### CohesiveAgentDeployment (initiating Asynchronous prediction for P2P)
### LSTM Architectures (Confidence calibration and provide short term memory)

## Main Component Deep-Dive

### 1. GeometricWeightShaping (AWE)
- **Purpose**: Analyzes data geometry to optimize weights
- **Input**: Raw data features
- **Output**: Optimized weight matrices
- **Key Algorithm**: Eigenvalue decomposition + spectral analysis

### 2. Specialized MLP
- **Input**: TF-IDF features
- **Hidden Layers**: Configurable (default 2-3)
- **Output**: Class probabilities
- **Advantage**: Noise-robust classification
- 
### 3. Optimized Transformer
- **Components**: Multi-head attention, feed-forward, layer norm
- **Key Innovation**: Alpha-based computing for stable training
- **Input**: Sequence embeddings
- **Output**: Contextual representations
- 
### 4. WeightedEnsemblePredictor
- **Combines**: MLP + Transformer predictions
- **Weighting**: Dynamic, based on attention quality + confidence
- **Methods**: Dynamic, Meta, or Calibration
- 
### 5. AgentDistributedInference
- **Components**: security handling, P2P handling and peer handling + socket
- **Output** Ensemble probabilities, P2P Coordination ouputs probability
### 6. Modelstorage

- **Components**: SQlite + json handling
- **Outputs**: Stored memory, attention weights, peer prediction.
  
### 7. ConsecutivePeerAgent
- **Components**: AgentDistributedInference + IntegratedPipeline + PipelinepredictionManager + socket
- **Outputs**: Predicted label from peer (notprobabilities)

### 8. CohesiveAgentDeployment
- **Components**: AgentDistributedInference + IntegratedPipeline + PipelinepredictionManager + socket
- **Outputs**: Predicted label from peer, probabilities, status of P2P.

### 9. LSTMCell
- **Components**: Contains LSTM and FF layer for initiating prediction and forward method.
- **Outputs**: Arrays of probabilities for LSTMNetwork

### 10. LSTMNetwork
- **Components**: Contains LSTMCell architecture model, and initiate training and Calibration method
- **Outputs**: Array of prediction

### 11. LSTMEngine
- **Components**: Contains LSTMnetwork architecture model, calibrate its prediction and confidence
- **Outputs**: Array of prediction, Confidence

## Data Flow intent

1. Raw Input → Text preprocessing
2. Encoding → TF-IDF + Sequence embeddings
3. Model Forward → MLP scores + Transformer context
4. Ensemble → Weighted combination
5. Calibration → Confidence adjustment
6. Memory Storage → SQLite persistence
7. SQlite persistence → Peer local coordination
8. P2P Components → Predicted label
   
 

