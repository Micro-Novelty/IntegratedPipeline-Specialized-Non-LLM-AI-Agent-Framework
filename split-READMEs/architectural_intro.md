## [+] MANN Intro
[=] Memory augmented Neural network (MANN) is a neural network architecture coupled with an external, dynamic memory module, allowing it to store, retrieve, and update information similarly to a computer's RAM. Unlike traditional networks that store knowledge only in weight parameters, MANNs excel at fast learning, long-term dependency handling, and episodic recall, In IntegratedPipeline, Its memory is stored in a custom database inside your local machine, then later used for memory retrieval, transfered to the AI Dictionary where it can finnaly recall its memory when input condition matched with memory. 

## [+] Abstract Weight Encoder (AWE) Intro
[=] AWE is a specialized custom weight shaping or encoding method Specifically designed for MLP Architecture, using eigenvalue and spectral methods as base equations to calculate covariance inside a given input data, and shape the correct Weight from the given eigenvalue, AWE Works by processing input and then captures the necessary eigenvalue to shape a properly initialized Weight that aligns with input data complexity, With this approach, MLP training will be much more consistent and robust against noise.

## [+] LSTM And Transformer Intro:
A. LSTMs (Long Short-Term Memory) and Transformers are foundational deep learning architectures built to process sequential data (like text or time series). While both handle the flow of time and context, they do so using completely different mechanisms.
- LSTM (Long Short-Term Memory):
  - LSTMs are an advanced class of Recurrent Neural Networks (RNNs) introduced to solve the problem of traditional RNNs forgetting earlier data.
  - How it works: LSTMs process data sequentially—one word or time-step at a time. They regulate information using "gates" (forget, input, and output) that determine what information from the sequence to keep or discard.
  - Use Cases: Ideal for tasks with strictly continuous chronological data like time-series forecasting (e.g., stock market or weather prediction) or speech recognition.
  - Limitations: Because they process data in a strict chain, it is difficult to parallelize training, making them slow and prone to forgetting long contexts.

B. Transformer:
Transformers are the modern standard for AI, introduced in 2017 with the famous "Attention Is All You Need" paper.
- How it works: Instead of reading sequences linearly, Transformers read the entire sequence all at once. They utilize a self-attention mechanism, which calculates how much "attention" or weight every part of the sequence should give to every other part, understanding the global context instantly.
- Use Cases: The backbone of Large Language Models (LLMs) like ChatGPT or BERT, making them perfect for machine translation, text generation, and summarization.
- Limitations: Transformers require massive amounts of training data and computing power to work effectively.
- Data Hungry when Analyzing Images.


[~] For a much In-Depth Explanation You can visit This repository to learn more about AWE and its performance results:
- Link: https://github.com/Micro-Novelty/Specialized-MLP-for-noise-robustness

## [+] Why IntegratedPipeline?
[~] IntegratedPipeline is a great choice for a sophisticated Non-LLM AI Program for The Main Orchestrator of a Distributed MANN-Type AI Agent Working in Edge-device/Consumer-Based machine Where LLM is'nt a great fit for Messy, Noisy environments. while still run efficiently on High-end Embedded systems in single-instance or as a distributed network during multi Agent cooperation.

[=] IntegratedPipeline offers:
1. Local-Based AI Orchestrator:
   - IntegratedPipeline Creates its own SQLite Database inside Your Computer once the library is executed, This database is used directly to store the AI Memory, Attention weights, predicted Output, and identified peer, all without leaving the machine, The Database will be created Automatically once you run the library, database name saved as activity_log.db. 

2. Continously Learning behavior for an Agent:
   - different from LLM that is static and cannot improve beyond its given training condition, AI Agent using IntegratedPipeline has a dynamic, flexible continously learning behavior with conditional training algorithms included in the library that has both supervised and unsupervised learning present, The learned input and predicted Output will be stored in the database, allowing it to recall its memory during processing and find matching known prediction given if input matched with the stored input inside the database. this Continous learning behavior is efficient because its not relying on weights for memory, allowing flexible and predictable behavior inside a given environment.
   
3. Robust Specialized MLP, Transformer and LSTM Architecture with ensemble weighting architecture:
  - IntegratedPipeline usually used 1 Model for regular prediction, but in some Conditions, it used 2 Different type's of AI Architecture stacked together, and one architecture to weight their confidence and probability fairly (ensemble method) to get the final prediction for the problem it faced, Specialized MLP for Noise robustness And Specialized Transformer that used Alpha-based Computing algorithm for contextual reasoning, LSTM architecture to provides proof-of-credibility over a certain output, acting as a support mechanism rather than Main orchestrator like MLP and Transformer. The reason why those Models complement and used together:
      - Specialized MLP Provides synchronous robust classification Against noise with its specialized Weight Encoder (AWE) to handle noise using eigenvalue based computing that is lightweight and efficient. This Method can't be replicated Inside Transformer FFN (Feed-forward-network) because of Transformer dynamic brute force computing where AWE-Based generated weight's get diluted over time by Transformer dynamic projection embedding, making AWE Generated weight causes inefficient inside Transformer dynamic FFN/QKV projection.
      - Specialized Transformer provides robust synchronous advanced contextual relationships, efficient data processing using Alpha based computing, The Transformer is tuned towards to be as flexible as possible to provide dynamic projection or fixed FFN projection training with minimal head's and dimension's to reduce computational power.
      - Our library also provides setup for kNN (k-Nearest-neighbor) Augmented Transformer for Users who Wants to use a much More Advanced Transformer, This Transformer is'nt used in the Main prediction flow inside IntegratedPipeline due to it being Memory heavy to run, instead this Transformer can be used separately if You have Larger Dataset and wants A small Transformer Model that can Remember longer contexts Using HNSW (Hierarchical Navigable Small World) Architecture and apply it on edge Devices.
        - Honest Limitation: Our Transformer can still analyze Images, by receiving Input that is an Image converted into X samples, its accuracy is expected to be much lower since Analyzing images requires larger datasets.
          
      - LSTM doesn't act as a Main orchestrator, instead it Provides coherent Short-term memory for the Ensemble architecture, acting as a support mechanism to provides proof-of-credibility of a given answer from past previous context input, this allows flexible and achievable Aggreement between Transformer and MLP over a short period of time.
      - Ensemble weighting provides the model a much more robust classification best from both worlds perspective, weighting both MLP and Transformer confidence and probability, combined with Attention quality from the transformer to get the final prediction of an input if transformer is allowed and permitted to be in use.
   

4. flexible and secure Peer-to-Peer Coordination (Multi-Agent):
   - IntegratedPipeline offers Peer to Peer communication capabilities asynchronously, Where IntegratedPipeline directly checks for other Peer presence directly to the local database present in the local computer or system (Synchronous prediction from peer previous data in the database), or externally, by using asynchronous request for initiating prediction, P2P is secured Using:
   - secure socket using user provided SSL CERT. on both client and server,
   - API key for requesting,
   - Alpha rate limiting,
   - HMAC secret key for authentication,
   - and IP validation.
   [~] Each agent has double roles during P2P:
        - Server provider: the peer Agent can start a server to listen for peer client's
        - Connecting Agent: the peer Agent which happens to connect to other peer that has or have opened and provided a server listener to act as a receiver.

5. Cross-Session memory availability:
   - IntegratedPipeline offers share-able Memory capability, included capability below:
     - Exportable memory: this allows a flexible memory saving for later use, such as cross transfer memory between model, the memory is saved as .json file after exporting.
     - Importable Memory: allowing to import memory from the exported .json file directly for the model to use.
     - syncing with other model: socket-based communication to export memory to other external machine.
       - [=] Note: socket Syncing is unsecure witout additional security layer wrapped, For a safer syncing, directly transfer the .json file memory to the target machine via other ways such as manual send.
     - list sessions: listing available sessions using model's memory name.

[=] With its Specialized Multi-Layer-Perceptron (using AWE Encoder) and Optimized Transformer module with optimized Embedding, IntegratedPipeline can directly tolerate low samples-amount of Data, including noisy ambiguous data, using Weighted Confidence assembling from both specialized MLP and Transformer for better reliability during training and prediction over Messy, noisy environments, such as:

[1.] User data's : User data is often messy and ambiguous, Specialized MLP with AWE will do the job for shaping the necessary Weight to complement for the Ambiguous noisy pattern, AWE MLP is highly robust to noise, proven in synthetic Environment such as scikit-learn Make-Classification scarce and Noisy Input robustness during generalization test. making it a great fit for messy, Scarce data.

[2.] Small Dataset's : We often don't have enough Dataset to train a Transformer Model, Thats Why IntegratedPipeline Offers a Highly optimized Transformer that supports scarce dataset processing, Using Alpha-Based computing as a Warm-up for training, it provides a direct Boost for the transformer to be efficient during training in scarce-data Environment.

[3.] Non-Representative data (Undersampled) : IntegratedPipeline Support's ambiguous data that come's from file with format such as CSV Format to extract title's and label's necessary to create automatic Dataset for Later use in Training, making it optimized for specific task's and easier dataset creation with lower overfitting rate for reliability.

_____________________________________________________________________________________________________________
[=] Architectural-Overview
<img width="1600" height="859" alt="New Arch" src="https://github.com/user-attachments/assets/a5db1ed2-9149-4635-8d95-d022ad6e8608" />
---
[=] Contextual meaning:
   1. - Sequence encoding is a machine learning technique that transforms a sequential input (like text, time-series data, or audio) into a compact, fixed-length numerical vector, often called a context vector
   2. - TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic used in machine learning and NLP to evaluate how important a word is to a document within a collection (corpus). It boosts rare words and penalizes common words (like "the", "and") by multiplying two metrics: how often a word appears in a document (TF) and the inverse frequency of the word across all documents (IDF). 
   3. - Explainability provides deeper transparency of why a model thought about a detail by showing its internal metrics like attention quality, from distributed peer memory or Ensemble prediction result's.


### Important Note:
- All of the files provided here can be found  in the github repository, if you are seeing this on PyPi and wanted to Try out the provided .py, .sh or Dockerfile scripts below, consider visiting the official github repository link Above.

### Introduction and demo
- Video Documentation: [![Introduction and demo:](https://youtube.com)](https://youtu.be/RmWvwDHU_QY?si=Lvl8mt8c_BnFypS_)
- Quick demo start: [main.py](main.py)
  - purpose: let you demonstrate the advanced prediction method and asynchronous prediction directly.
- Quick test of P2P:
    - [multi_agent_client.py](P2P_Setups/multi_agent_client.py)
    - [multi_agent_server.py](P2P_Setups/multi_agent_server.py)
    - [P2PDirectTest.py](P2P_Setups/P2PDirectTest.py)
    - Purpose: let you demonstrate simple P2P using AbstractIntegratedModule quickly, there may be bugs in this P2P setup so feel free to share it in issues.
