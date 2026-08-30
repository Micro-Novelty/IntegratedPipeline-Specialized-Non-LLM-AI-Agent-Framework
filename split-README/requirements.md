____________________________________________________________________________________________________________________
## [=] Requirements
[~] To run and execute IntegratedPipeline, Requirement's include Or see [Requirements](installation_guides/usage_needs.txt):
- pip install for quick usage:
- ```bash
  pip install AbstractIntegratedModule # or
  python -m pip install AbstractIntegratedModule
  ```
  
- (Optional) Download AbstractIntegratedModule Compiled Binary file extension manually in the repository for Users who can't use pip install (Choose one minimal, specified for your needs):
   - AbstractOptimizedModules for Optimizing Transformer using Cython (Optional)
   - AbstractIntegratedModule (Main library)
   - Compiled Binaries can be downloaded from release section in the Main repo or You can use this link:
     - Release section: https://github.com/Micro-Novelty/IntegratedPipeline-Specialized-Non-LLM-AI-Agent-Framework/releases
     - Supported OS: Mac-Os, linux x86-64 and Windows.
  
- NOTE: for Mac-OS or linux x86-64 Users, consider downloading the correct python wheel for your setup in here for a much flexible installation:
   - ```bash
     pip install abstractintegratedmodule --extra-index-url https://Micro-Novelty.github.io/abstract-modules/whl/ --break-system-packages
     ```
     Note: using this installation method grants you a much safer, predictable behavior, and more secure AbstractIntegratedModule library rather than a pure binary file.
  
   - (Optional) Libraries needed (For using Compiled binary, not python wheels):
     - Pandas
     - aiohttp
     - scikit-learn
     - numpy
     - psutil
     - cryptography
   - (Optional) Install AbstractIntegratedModule Compiled binaries from release section:
  
[=] for labels assignation:
- CSV file that contains training labels and titles used for training and prediction label map, example will be provided below, Go to Step's for in-depth Usage below.

_______________________________________________________________________________________________________________________
### [=] Requirements for Docker container
- Dockerfile (For Container assembler) 
- Python scripts (Such as main.py for Dockerfile usage).
- entrypoint.sh (for smart entry point for Dockerfile container usage).
  
## [=] System-Specific Notes
1. Windows:
   - Requires Visual C++ Build Tools for compatibility
   - Use PowerShell or CMD (not WSL bash for best results)
     
2. Linux:
   - Ensure gcc and build-essential are installed
   - Different distributions may require different package managers
     
3. ARM64 - Raspberry Pi
   - Installation may take 30+ minutes due to ARM architecture
   - Monitor system resources during installation
   - Consider using faster storage (USB SSD) for better performance

## [=] Docker Container Application
0. See [Docker_installation_Section](docker-setups/Docker_installation_Section.md) for an in-depth explanation, or [Quick_Docker_start](docker-setups/Quick_Docker_start.sh) for a quick start.
   - Note Consider checking:
     - [Dockerfile](docker-setups/Dockerfile) contains all the instructions need to assemble a Docker container.
     - [start.sh](docker-setups/start.sh) for Quick single agent start in Docker container.
     - [entrypoint.sh](docker-setups/entrypoint.sh) for a Smart entrypoint used in Dockerfile (Must be downloaded too along with Dockerfile and put in the same folder with Dockerfile after its downloaded)
     - [start-multi-agent-cluster.sh](P2P_Setups/start-multi-agent-cluster.sh) for Multi-agent cluster start in Docker container, What it does:
       - Starts a multi-agent Docker cluster — runs docker-compose up -d in detached mode (background), scaling the agent-client service to 5 simultaneous instances, alongside whatever server is defined in the docker-compose.yml.
       - Tails the logs — runs docker-compose logs -f which streams live logs from all containers (server + all 5 clients) to your terminal until you hit Ctrl+C.
     - [main.py](main.py) for executing a python script in the Docker container that used main.py, like in this code:
        - ```bash
          docker run -it -v $(pwd):/app/data integrated-pipeline:latest python /app/data/main.py
          ```
     - To use a [.dockerignore](docker-setups/.dockerignore) file, place it in your build context directory (the same location as your Dockerfile) to specify which files and folders should be excluded when building your image.
     - [.env](.env) is used for environment setup in [entrypoint.sh](docker-setups/entrypoint.sh), this file must be in the same folder with Dockerfile and entrypoint.sh after its downloaded.

1. Build Image:
   - Clone repository:
   - ```bash
     git clone https://github.com/Micro-Novelty/IntegratedPipeline-Continous-Learning-AI-Agent-library-framework.git
     cd IntegratedPipeline-Continous-Learning-AI-Agent-library-framework
     ```
   - Download:
   - [Dockerfile](docker-setups/Dockerfile),
   - [entrypoint.sh](docker-setups/entrypoint.sh)
   - [.env](docker-setups/.env)
   - In the code or release section.
     
     - If the downloaded Dockerfile or .dockerignore or .env has .txt extension, remove the extension:
     - ```bash
       # remove .txt extension
       mv Dockerfile.txt Dockerfile
       mv env.txt .env
       mv dockerignore.txt .dockerignore
       ```
   - Navigate to the folder: Use the cd command to enter the directory containing the Dockerfile and entrypoint.sh file.
   - ```
     cd /path/to/your/folder
     ```   
   - build image:
   - ```bash
     sudo docker build -t integrated-agent.
     ```
2. Download library dependencies for binaries usage:
   - [Optional]: Required libraries (when using binary, not python wheel provided in the library):
   - Numpy
   - Scikit-learn
   - pandas
   - aiohttp
   - psutil
     
4. Run IntegratedPipeline in a Container:
    - Install AbstractIntegratedModule via PIP or Compiled binaries:
    - ```bash
      pip install AbstractIntegratedModule # or
      python -m pip install AbstractIntegratedModule
      ```
      
    - ```bash
      docker run -it --name ai-agent integrated-agent:latest python
      ```
   - In python shell:
       - ```
         from AbstractIntegratedModule import IntegratedPipeline, PipelinePredictionManager
         model = IntegratedPipeline('agent_memory')
         print("✓ IntegratedPipeline initialized successfully!")
         ```
         
5. Run script:
   ```bash
   # Mount your local directory and run a script
   docker run -it -v $(pwd)/data:/app/data integrated-agent:latest python main.py # main.py could be replaced
   ```
   
6. Run with GPU Support (Optional):
   - ```bash
     # For NVIDIA GPU support
     docker run -it --gpus all -v $(pwd)/data:/app/data integrated-agent:latest python main.py
     ```
     
7. For Single Agent and Multi-Agent P2P:
   
   [=] Single agent:
   ```bash
   # Build image
   docker build -t integrated-agent:latest .

   # Run single agent
   docker run -it -v $(pwd)/data:/app/data integrated-agent:latest python
   ```
   
   [=] Multi agent P2P (Consider docker-compose) :
   - Note: Use the provided [docker-compose.yml](docker-setups/docker-compose.yml) for Quick multi-agent.
   - Navigate to the folder: Use the cd command to enter the directory containing the docker-compose.yml file.
   - ```
     cd /path/to/your/folder
     ```
   - run this command to run docker-compose.yml:
   - ```bash
     # Start multiple agents
     docker-compose up -d

     # View logs
     docker-compose logs -f

     # Stop all agents
     docker-compose down
     ```

________________________________________________________________________________________________________________________
## Performance in linux ARM64 Docker Environment/Container
A. [=] Computational performance results with Transformer included during Advanced prediction method.
```txt
== TIME == | CPU % | RAM / RAM LIMIT ||

14:50:08.762  0.00%  25.52MiB / 3.71GiB 
14:50:10.339  5.56%  25.52MiB / 3.71GiB
14:50:12.428  0.17%  25.52MiB / 3.71GiB
14:50:15.835 103.60% 38.8MiB / 3.71GiB
14:50:17.844 438.11% 60.15MiB / 3.71GiB
14:50:19.896 419.36% 68.32MiB / 3.71GiB
14:50:22.022 434.15% 73.2MiB  / 3.71GiB
14:50:23.969 436.47% 78.18MiB / 3.71GiB
14:50:25.966 435.85% 80.22MiB / 3.71GiB
14:50:28.018 430.47% 92.12MiB / 3.71GiB
14:50:30.029 450.55% 99.5MiB / 3.71GiB
14:50:32.065 438.06% 105.4MiB / 3.71GiB
14:50:34.100 469.52% 115.2MiB / 3.71GiB
14:50:36.109 449.89% 116.8MiB / 3.71GiB
14:50:38.123 423.56% 119MiB   / 3.71GiB
```
- [=] Note: To read the full logs (For included Transformer performance): [performance_log_with_TF](performance_logs/performance_log_with_TF.txt)
- [=] Explanation: 
  - Average CPU Usage = 420% - 430% , Sustained roughlt 4.20 - 4.30 CPU Cores
    - Meaning = - Compute-Intensive workload
                - reduced guaranteed of experiencing deadlocks, thread starvation, and major stalls
                - numerical backend is parallelizing efficiently
  - Average RAM Usage = 180 - 195 Mib, Roughly 4.5 - 5%  usage of available RAM
    - Meaning = - Very efficient RAM Usage
                - Repeated 269 MiB suggests allocator settled, ndarray pools stabilized, workload reached steady state
                - gradual, safe and steady batch accumulation, reduced possibility of leak explosion.

B. [=] Advanced Prediction without Transformer, Only Specialized MLP + LSTM using AWE.
```txt
== TIME == | CPU %  | RAM \ RAM LIMIT |
10:01:37.456 100.44% 269.8MiB / 3.71GiB # initial round 1 training started
10:01:38.686 101.93% 279.8MiB / 3.71GiB
10:01:40.694 101.94% 293.2MiB / 3.71GiB
10:01:42.703 99.90%  302.9MiB / 3.71GiB
10:01:44.712 99.34%  319.8MiB / 3.71GiB
10:01:46.721 100.15% 319.7MiB / 3.71GiB
10:01:48.727 101.91% 319.7MiB / 3.71GiB
10:01:50.734 100.86% 319.7MiB / 3.71GiB
10:01:52.742 136.43% 322.9MiB / 3.71GiB
10:01:54.751 101.91% 324.9MiB / 3.71GiB
10:01:56.758 100.29% 324.9MiB / 3.71GiB
10:01:58.769 102.42% 327.1MiB / 3.71GiB
10:02:00.775 0.00%   329.6MiB / 3.71GiB # First Training and prediction round 1 finished
10:02:02.788 19.55%  330.1MiB / 3.71GiB # round 2 training
10:02:04.798 134.23% 332.2MiB / 3.71GiB
10:02:06.803 108.80% 332.2MiB / 3.71GiB
10:02:08.811 107.85% 332.2MiB / 3.71GiB
10:02:10.857 104.43% 332.2MiB / 3.71GiB
10:02:12.854 101.14% 332.2MiB / 3.71GiB
10:02:14.858 0.00%  332.3MiB / 3.71GiB
10:02:16.873 98.84% 333.8MiB / 3.71GiB
10:02:18.869 0.00%  132.7MiB / 3.71GiB # container stopped and round 2 finished
```
- [=] Note: See full performance log in here: [performance_log_only_MLP](performance_logs/performance_log_only_MLP.txt)
- [=] Explanation:
     - Average CPU Usage = 85-90% Used, sustained roughly 1 CPU Core saturated on average
         - Meaning: - Very efficient computation
                 - Much lighter CPU Usage than Transformer.
                 - Stable and less parallelizing behavior (Efficient computation)    
                 - 4.7x lower CPU usage than transformer included
      - Average RAM usage = 314 MiB / 3.71 GiB, Roughly 8.5% usage of available RAM.
         - Meaning: - Slightly larger MiB used than Transformer.
                 - Model allocation growth is stable,
                 - training buffers stable
                 - temporary ndarray growth.

### Both performance Overview
<img width="1536" height="1024" alt="WhatsApp Image 2026-05-24 at 10 27 00" src="https://github.com/user-attachments/assets/9404277f-281f-4893-8367-e494833230ea" />
_______________________________________
- Note: 
   - This performance metric was calculated in older versions, With newer Transformers that has more, newer modules, The resulting cpu consumed may Increase 1.5 - 2x much more in Docker ARM64 Environment + QEMU, 
   MLP And LSTM stays the same and still fully optimized.

## Source code of AbstractIntegratedModule 
- Note: The source code is provided in the repository.
- [=] Full Monolithic extensively-documented source code (12K+ Lines): [AbstractIntegratedModule.py](src/AbstractIntegratedModule.py)
- [=] Separated Modules of AbstractIntegratedModule With proper Documentation: [separated_module](separated_module)
- [=] up-to-date Source code: [AbstractIntegratedModule.pyx](src/AbstractIntegratedModule.pyx) and [AbstractOptimizedModules.pyx](src/AbstractOptimizedModules.pyx) files for independent compilation or direct compiling on real ARM64 device or if you want to see the up-to-date, undocumented version of AbstractIntegratedModule library.
- [lib.rs](rust_optimization_setups/lib.rs) that contains Rust optimization for Models weight handling.
- [~] Note:
   - The source code is open and Free to anyone who:
   - Use it: Run the software for any personal, academic, or commercial purpose.
   - Modify it: Change the source code to fit their needs.
   - Distribute it: Share the original or modified code with others.
   - Commercialize it: Package, brand, and sell the software for profit.
