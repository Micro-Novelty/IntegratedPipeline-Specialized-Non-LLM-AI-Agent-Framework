# [=] AbstractIntegratedModule, Specialized-AI-Agent-framework for Non-LLM Continual-learning edge AI Models.

[~] Introduction:
- AbstractIntegratedModule or IntegratedPipeline in short, is a standalone Specialized AI Agent Library for Non-LLM memory Augmented Agentic Framework orchestrator - Specifically designed to provide Agentic capability for any Autonomous Agentic Framework locally and Coordinatively that runs efficiently from consumer based machine to High-end embedded systems, where the AI Can directly and continously learn, with minimal and efficient compute, built-in augmented memory, Secure Peer-To-Peer (Multi-Agent) Coordination with security layers as an option, And Explainability capability based on proof from in it's internal metrics, reducing Black-Box condition necessary for reliability. 
- AbstractIntegratedModule Contains specialized MLP using Its Own specialized geometric Weight shaping (AWE), Specialized efficient Transformer and LSTM (Long-short term memory) architecture for Scarce Data with Alpha-based computation, specifically designed for low-amount samples environment or Messy environments.
  - IntegratedPipeline Use-Case:
    - Tabular Data Prediction: Finding patterns in structured spreadsheets or rows of numerical and categorical features.
    - Classification Tasks: Separating data into groups, such as spam detection, customer churn, or disease diagnosis.
    - Regression Tasks: Predicting continuous numerical values like house prices, sensors output, sales numbers, or weather metrics.

___________________________________________________________________________________________________________________
### [+] Framework's Purpose
  - This framework is intended for Engineers who is early in ML Engineering in order to provide a Clearer picture of How edge AI Should operate, Low memory management, and Also how it can be Made continuously learn using Pragmatic solution by using SQLite to save Weights to prevent continuous catasthropic forgetting, IntegratedPipeline source code is open-sourced and can be found in our github repository, Meaning you can modify the Source code however You liked it to be, and help Us Grow the first Community!
    
____________________________________________________________________________________________________________________
### Library Short Description
- Development Stage on PyPi: 1.2.0 Official Release.
- Author and Maintainer: Micro-Novelty and EpsitronNet-bot.
- library Source-Code is Open-sourced with MIT License.
- Purpose: Specifically Designed for providing Non-LLM AI Agent Framework for edge Devices, Optimized for ARM64 architecture.
- Library installation: 
  ```bash
  pip install AbstractIntegratedModule
  python -m install AbstractIntegratedModule
  ```

### Github Link (for Visiting and cloning)
- https://github.com/Micro-Novelty/IntegratedPipeline-Specialized-Non-LLM-AI-Agent-Framework

- The library also includes precompiled binaries for:
 - aarch64 manylinux (accepts version 2.17+) architecture, accepts python version 3.10, 3.11 only.
 - aarch64 musllinux (accepts version 1.2+) architecture, accepts python version 3.10, 3.11 only.
 - Windows 64 bit architecture (python 3.10, 3.11, 3.12, 3.13 only)
 - single tarball file for Users who want to test the library up to Python version 3.12 or 3.13+.
 ____________________________________________________________________________________________________
- Library installation if you dont have aarch64 setup, you can download the correct wheel for your setup in this repository or by using pip:
  - ✨ use pip for downloading the correct wheels for your setup:
     - ```bash
       pip install abstractintegratedmodule --extra-index-url https://Micro-Novelty.github.io/abstract-modules/whl/ --break-system-packages
       # ensures proper installation by bypassing pip strict external download setup. 
       ```
       - Note: Using the above installation --extra-index method provides Python Wheels for:
       - x86_64 and aarch64 with manylinux (v. 2.17+) and musllinux (v. 1.2+) architecture.
       - macOS with aarch64 and x86_64 architecture. (v. 10.9+)
       - All of this Wheels provided in the Repository Only Provides Wheels for python with version 3.10 3.11 and 3.12.
       - This setup will automatically download the correct wheel Based on your python/pip version and OS / hardware architecture setup.

- For specific module in Rust for handling and loading Weights and Parsing JSON values with reduced memory lookup overhead.
 - Consider using this Optimization library by using:
    - ```bash
      pip install abstract-weights-core
      ```
    - Note: This Optimization would help AbstractIntegratedModule with faster JSON parsing and much more flexible database handling handled efficiently using Rust programming language.
   
- Proven Capabilities:
   - The library has been thoroughly tested in Multiple Environments from Windows to ARM64 Environment. The library is now Robust for Wider use and Deployment.
   - Proven Works on ARM64 Environment, Training and Prediction works efficient on Docker ARM64 environment with QEMU, good parallelizing behavior is guaranteed.
   - P2P Works efficiently in ARM64 Docker + QEMU, No conflicting socket and all prediction works efficiently.
   - AWE setup Proven Efficient on Hard-uncontrolled dataset such as Activity Recognition from the given Database.
   - LSTM is Optimized efficiently for scarce data with AWE method.
   - Robust Advanced prediction capabilities proven effective on ARM64 Using MLP + LSTM Architectures.
   - Transformer Optimized using Cython, reduced Memory overhead and Reduce CPU Usage, With Reduced Training Time is guaranteed.
-----
  - Changelog:
     - v1.2.0:
        - [=] New features:
        - Added new Architecture module separate from Main Pipeline usage (This modules is'nt used inside Pipeline prediction):
          - Small-HNSW (Hierarchical Navigable Small World).
          - kNN-Augmented Transformer.
          - PerHeadMemory class to Apply memory to each Transformers Head.
        - Added new Functions to call the kNN-Augmented Transformer for separate usage.
            
          
     - Note: if you want to see the Changelog history of the library Older versions consider visiting this link:
       - PyPi history: https://pypi.org/project/AbstractIntegratedModule/#history
       
________________________________________________________________________________________________________________________
### [>] Full Documentation Features
- [Go to IntegratedPipeline-Specialized-AI-Agent-library](#[=]-abstractintegratedmodule-specialized-ai-agent-framework-for-non-llm-continual-learning-edge-ai-models)
- [Go to Framework purpose](#framework's-purpose)
- [Go to Library short description](#library-short-description)
- [Go to MANN Intro](#-MANN-Intro)
- [Go to Performance Overview](#both-performance-overview)
- [Go to Abstract Weight Encoder (AWE) Intro](#-abstract-weight-encoder-awe-intro)
- [Go to LSTM And Transformer Intro](#-lstm-and-transformer-intro)
- [Go to Why IntegratedPipeline?](#-why-integratedpipeline)
- [Go to Requirements](#-requirements)
- [Go to System-Specific-Notes](#-system-specific-notes)
- [Go to Quickstart with Docker](#-docker-container-application)
- [Go to Performance in ARM64 Environment/Container](#performance-in-linux-arm64-docker-environmentcontainer)
- [Go to Step's to use IntegratedPipeline](#-steps-for-library-in-depth-usage)
- [Go to Troubleshooting](#-troubleshooting)
- [Go to Detailed process of Alpha-computing](#-detailed-process-of-alpha-computing)
- [Go to Main Components](#main-components)
- [Go to Source code](#source-code-of-abstractintegratedmodule)

Consider checking:
  - [ROADMAP.md](ROADMAP.md)
  - [SECURITY.md](SECURITY.md)
  - [Contributing.md](Contributing.md)
  - [changelog.md](changelog.md)
  - [requirements-For-Dev](dev_needs.txt) for contributors requirements.
  - [architecture_diagram.js](architecture_diagram.js).
    
