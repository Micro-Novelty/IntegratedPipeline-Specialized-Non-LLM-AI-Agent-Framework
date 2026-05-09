flowchart TD
    A[Input Text] --> B[TF-IDF Vectorizer]
    B --> C[AWE MLP]
    B --> D[Transformer]
    
    C --> E[Weighted Ensemble]
    D --> E
    
    E --> F[Memory Gate]
    F -->|Cache Hit| G[Return Cached]
    F -->|Cache Miss| H[Compute Fresh]
    
    H --> I[Explainability]
    I --> J[Storage]
    J --> K[Output]
    
    G --> K
    
    subgraph "Optional"
        L[Distributed Peer] -.-> E
    end
