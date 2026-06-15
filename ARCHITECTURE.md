# Architecture Documentation

## System Overview

IntegratedPipeline consists of 20 interconnected components working together.

## High-Level Integrated Architecture
### IntegratedPipeline
### WeightedEnsemblePredictor (Ensemble weighting)
### AgentDistributedInference (P2P handling)
### Model storage (Database handling)
### ConsecutivePeerAgent (Ensemble P2P Handling)
### CohesiveAgentDeployment (initiating Asynchronous prediction for P2P)
### LSTM Engine/LSTM Network and LSTM Cell

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

### 9. LSTM Engine:
- **Components*: AWE method, AME encoder, LSTM Cell + LSTM network
- **Outputs**:Arrays of probs, Confidence, samples, intervals.

## Data Flow intent

1. Raw Input → Text preprocessing
2. Encoding → TF-IDF + Sequence embeddings
3. Model Forward → MLP scores + Transformer context
4. LSTM Architectures -> Ensemble
5. Ensemble → Weighted combination
6. Calibration → Confidence adjustment
7. Memory Storage → SQLite persistence
8. SQlite persistence → Peer local coordination
9. P2P Components → Predicted label
   
 

