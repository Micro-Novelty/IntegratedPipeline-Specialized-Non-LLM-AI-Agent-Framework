# IntegratedPipeline---Custom-AI-Agent-Core-library

[~] IntegratedPipeline is a standalone Custom AI Agent Library for memory Augmented Agentic Framework, Specifically designed to provide Agentic capability for any Autonomous Agentic Framework locally and Coordinatively that runs efficiently on High-end embedded systems, where the AI Can directly and continously learn data's with minimal compute, with augmented memory init, Secure Peer-To-Peer Sharing with ssl as an option, And Explainability capability based on proof from in it's internal metrics, reducing Black-Box condition necessary for reliability. Containing specialized MLP using Its Own specialized geometric Weight shaping (AWE) and Specialized Transformer for Scarce Data.

<img width="393" height="385" alt="1000077388-removebg-preview" src="https://github.com/user-attachments/assets/c7794da0-f9c5-4c61-8b63-642700b965f5" />

# MANN Intro:
[=] Memory augmented Neural network (MANN) is a neural network architecture coupled with an external, dynamic memory module, allowing it to store, retrieve, and update information similarly to a computer's RAM. Unlike traditional networks that store knowledge only in weight parameters, MANNs excel at fast learning, long-term dependency handling, and episodic recall, In IntegratedPipeline, Its memory is stored in a custom database inside your local machine, then later used for memory retrieval, transfered to the AI Dictionary where it can finnaly recall its memory when input condition matched with memory.

# Abstract Weight Encoder (AWE) Intro:
[=] AWE is a specialized custom weight shaping method that used eigenvalue and spectral methods to calculate covariance inside a given input data, and shape the correct Weight from the given eigenvalue, AWE Works by processing input and then captures the necessary eigenvalue to shape a properly initialized Weight that aligns with input data complexity, So, MLP training will be much more consistent and robust against noise.

[~] For a much In-Depth Explanation You can visit This repository to learn more about AWE and its performance results:
- Link: https://github.com/Micro-Novelty/Specialized-MLP-for-noise-robustness

# Why IntegratedPipeline?
[~] IntegratedPipeline is a great choice for deploying Edge-device AI Agentic framework and High-end Embedded systems, With its Custom Research-Grade Multi-Layer-Perceptron (AWE) and Enhanced Transformer Embedding that can directly tolerate scarce Data using Weighted Confidence assembling for better reliability over Messy environments, such as:

[1.] User data's : User data is often messy and ambiguous, The Specialized MLP will do the job for shaping the necessary Weight to complement for the Ambiguous noisy pattern, AWE MLP is highly robust to noise, proven in synthetic Environment such as scikit-learn Make-Classification scarce and Noisy Input robustness during generalization test. making it a great fit for messy, Scarce data.
[2.] Small Dataset's : We often don't have enough Dataset to train a Transformer Model, Thats Why IntegratedPipeline Offers a Highly optimized Transformer that supports scarce dataset processing, Using Alpha-Based computing as a Warm-up for training, it provides a direct Boost for the transformer to be efficient in scarce-data Environment. 
[3.] Non-Representative data (Undersampled) : IntegratedPipeline Support's Large ambiguous data that come's from file with format such as CSV Format to extract title's and label's necessary to create automatic Dataset for Later Training from the given dat'as, making it optimized for specific task's and easier dataset creation with lower overfitting rate for reliability.

# Requirements:
[~] To Quickly Run IntegratedPipeline, Requirements for it include's:
- Windows Native OS 
- Linux x86_64 and Raspberry Pi
- Python 3.13+
- AbstractIntegratedModule.pyd (Main Module)
- AbstractIntegratedModule.cpython-39-x86_64-linux-gnu.so (For linux x86_64)
- AbstractIntegratedModule.cpython-39-aarch64-linux-gnu.so (for Linux ARM64 - Raspberry Pi)
- CSV file that contains training labels and titles.

# Steps for Usage:
1. Download:
   - AbstractIntegratedModule.pyd (For Windows),
   - AbstractIntegratedModule.cpython-39-x86_64-linux-gnu.so (For linux x86_64) 
   - AbstractIntegratedModule.cpython-39-aarch64-linux-gnu.so (for Linux ARM64 - Raspberry Pi)
     
2. Create CSV file that contains training labels and titles:
   -  Example format:
      ```
      window_title, label
      "Thesis.docx", focused_work,high,writing thesis
      "Microsoft Excel", work,medium,data analysis
      "YouTube - Google Chrome", distracted,high,watching videos
      "Slack", communication,high,team chat
      "VSCode", focused_work,high,coding
      "netflix.com - Google Chrome", break,high,Netflix break
      "Outlook", work,medium,checking email
      "System Settings", system designing,low,configuring computer
      "GitHub", creating and editing repo, research
      "README.md - VS Code", focused_work,medium,reading docs
      "Amazon.com - Chrome", personal work,high,shopping
      ```
      Note = window_title is target_title and label is target_label, check step below to use it.

4. Use IntegratedPipeline as in this example:
   ```
   from AbstractIntegratedModule import IntegratedPipeline
   from AbstractIntegratedModule import PipelinePredictionManager

   memory_name = 'agent_memory'
   main_model = IntegratedPipeline(memory_name, ssl_cert_file=cert_file, ssl_key_file=key_file) # provide cert_file path or key_file path (optional)
   main_prediction = PipelinePredictionManager(main_model, label_csv='example_manual_training.txt', target_title='window_title', label='label')
   # example_manual_training is a .txt file that contain csv format like above example.
   
   examples_rules = [
                        # === WORK / PRODUCTIVITY ===
                        (r'code|programming|develop|debug|compile|script', 'focused_work'),
                        (r'vscode|visual_studio|ide|terminal|shell', 'focused_work'),
                        (r'notion|evernote|onenote|notes|todo|task', 'productive'),
                        (r'slack|teams|discord|zoom|meeting|call', 'communication'),
                        (r'email|gmail|outlook|inbox|mail', 'communication'),
                        
                        # === ENTERTAINMENT ===
                        (r'youtube|netflix|twitch|stream|video', 'entertainment'),
                        (r'music|spotify|soundcloud|audio|player', 'entertainment'),
                        (r'game|gaming|steam|epic|play', 'gaming'),
                        (r'facebook|instagram|tiktok|social|post', 'social_media'),
                        
                        # === BROWSING ===
                        (r'chrome|firefox|edge|safari|browser', 'browsing'),
                        (r'google|search|wiki|wiki|article', 'information'),
                        (r'stackoverflow|github|docs|documentation', 'research'),
                    ]
                        
   titles, _, label_map = main_prediction.load_labels_from_csv(<your_filename>, <target_title>, <target_label>)
   datasets, X = main_model.data_preparation(titles, label_map)
   results, chosen_label, confidence = main_model.advanced_prediction_method(self, titles, label_map, example_rules,
                                show_proba=False, top_k=3, 
                                use_transformer=True,
                                return_attention=False,
                                save_results=True)

   # ... more features you can add
   ```
   5. As an option, You can add more feature's directly to what it should predict, behave using rules you have given, Create a visual dashboard, and much more.

   

# Main Components:
[=] With a total of 17 different stacked Architectures, The main Component's of IntegratedPipeline is:

1. GeometricWeightShaping
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
---
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







