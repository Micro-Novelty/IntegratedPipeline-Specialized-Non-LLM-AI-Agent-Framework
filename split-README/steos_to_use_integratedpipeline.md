______________________________________________________________________________________________________________________
## [=] Step's for library in-depth Usage
0. Download via PIP:
   - Clone repository first:
     ```bash
      # Clone immediately for Windows and x86_64 only without prerequisites          
      git clone https://github.com/Micro-Novelty/IntegratedPipeline-Continous-Learning-AI-Agent-library-framework.git
      cd IntegratedPipeline-Continous-Learning-AI-Agent-library-framework     
      ```

   - Install the library via pip:
     ```bash
     pip install AbstractIntegratedModule #or
     python -m pip install AbstractIntegratedModule
     ```

1. Clone repository:

 - ```
   # prerequisites (for Raspberry pi OS Only)
   # Update system
   sudo apt-get update
   sudo apt-get upgrade -y

   # Install Python 3.x and development tools
   # Version 3.x means you can install python 3.10 to 3.13 only,
   # choose one version specified for your needs (3.10 or 3.11 or 3.12 ...).
   sudo apt-get install python3.x python3.x-dev python3.x-venv -y

   # Install additional build tools
   sudo apt-get install build-essential libatlas-base-dev libjasper-dev -y

   # Clone immediately for Windows and x86_64 only without prerequisites          
   git clone https://github.com/Micro-Novelty/IntegratedPipeline-Continous-Learning-AI-Agent-library-framework.git
   cd IntegratedPipeline-Continous-Learning-AI-Agent-library-framework     
   ```   
 2. Install System Dependencies (for x86_64 installation):
    ```
    # Ubuntu/Debian
    # example with Version 3.10:
    sudo apt-get update
    sudo apt-get install python3.10 python3.10-dev python3.10-venv

    # CentOS/RHEL
    sudo yum install python310 python310-devel

    # Fedora
    sudo dnf install python3.10 python3.10-devel
    ```
    
 3. Create a virtual environment:
     - ```
       # Create virtual environment (windows)
       # example with python 3.10
       python -m venv venv
       # Activate virtual environment
       venv\Scripts\activate
       
       # Create virtual environment (x86_64) (ARM64 / raspberry pi)
       python3.10 -m venv venv
       # Activate virtual environment
       source venv/bin/activate
       ```
         
   
 4. Verify Installation:
    - ```
      python -c "from AbstractIntegratedModule import IntegratedPipeline; print('✓ Installation successful!')"
      ```
      
 5. Run main.py for quick test of successful imports:
    - Download main.py in our repository code section: [main.py](main.py)
    - ```
      # run this for quick import test.
      python main.py
      ```
           
         
     
3. Create CSV file that contains training labels and titles (Optional for training using texts):
   -  Example format:
      ```txt
      window_title,label
      "Thesis.docx",focused_work,high,writing-thesis
      "Microsoft Excel",work,medium,data-analysis
      "YouTube -> Google Chrome",distracted,high,watching-videos
      "Slack",communication,high,team-chat
      "VSCode", focused_work,high,coding
      "netflix.com -> Google Chrome",break,high,Netflix-break
      "Outlook",work,medium,checking-email
      "System Settings",system designing,low,configuring-computer
      "GitHub",creating-and-editing-repo,research
      "README.md -> VS Code",focused_work,medium,reading-docs
      "Amazon.com -> Chrome",personal-work,high,shopping
      ```
      Note = window_title is target_title and label is target_label, check step below to use it.


4. A. Use IntegratedPipeline as in this example:
   ```python
   from AbstractIntegratedModule import IntegratedPipeline
   from AbstractIntegratedModule import PipelinePredictionManager
   import numpy as np

   # SETUP BLOCK.
   memory_name = 'agent_memory'
   cert_file = <your_cert_file_dir> # your .crt file
   key_file = <your_key_file_dir> # your .key file

   # SSL Setup for users who used Lets encrypt / Public CA cert:
   # Server context
   server_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
   server_ctx.load_cert_chain('your_server.crt', 'your_server.key') # No load_verify_locations needed — OS trust store handles public CA
   
   # Client context  
   client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
   client_ctx.load_cert_chain('your_client.crt', 'your_client.key') # no load_verify_locations — OS trust store handles it

   # for enterprise and internal CA:
   '''
   server_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
   server_ctx.load_cert_chain('their.crt', 'their.key')
   server_ctx.load_verify_locations('company_ca.crt')  # internal CA not in OS store
   server_ctx.verify_mode   = ssl.CERT_REQUIRED
   server_ctx.check_hostname = False
   
   client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
   client_ctx.load_cert_chain('their.crt', 'their.key')
   client_ctx.load_verify_locations('company_ca.crt')
   client_ctx.verify_mode   = ssl.CERT_REQUIRED
   client_ctx.check_hostname = False
   ''' # uncomment to use

   # Note: The Above setup is Important for Users who wants to do Secure P2P In deployment case, for local P2P the above setup is optional and automatic self-signed CERT will be used for local-device P2P.

   main_model = IntegratedPipeline(
      memory_name=memory_name,  # memory name for the AI you already initialized
      use_async=True, # local asynchronous prediction is permitted, if not PipelineAsyncManager wont start asynchronous prediction.
      agent_port=5001, # this port is used to set AgentDistributedInference server (optional)
      ssl_cert_file=cert_file, ssl_key_file=key_file,# provide your cert_file path or key_file path (optional)
      ssl_context=server_ctx, # used by the Agent server. (optional)
      client_ssl_context=client_ctx # used by the client. (optional)
      ) 

   main_prediction = PipelinePredictionManager(
      main_model, # your initialized pipeline
      label_csv='example_manual_training.txt', 
      # your filename that contains the .txt file and contains the CSV format.
      # the Agent will automatically searched the nearby folder like: downloads, data, and desktop folder.
      target_title='window_title', label='label')

   # example_manual_training is a .txt file that contain csv format like above example.

   # rules will be used to create automatic dataset for IntegratedPipeline.
   example_rules = [
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

                        # more rules
                    ]

   # activate explainability capability to explain uncertainty:
   main_model.show_explainability_details = True
   main_model.distribution.predict_manager = main_prediction # set PipelinePredictionManager to AgentDistributedInference for asynchronous prediction later (Very important for asynchronous prediction)
   # main_model.use_transformer = True if you want to use transformer, this will notify all modules that used advanced_prediction_method will initiate prediction with both transformer and MLP.

   # set IntegratedPipeline Penalty rate when it output wrong answer:
   main_model.error_decay = 0.75
   # error_rate > 0.5 means old errors fade quickly — a class that was wrong 3 predictions ago matters less than one wrong just now, making the model less likely to output repetitive wrong answer.
   # this a flexible tunable-knob for the model judgement regarding wrong answer, this will propagate through prediction layers to inform about the model repetitive answer and calibrate it immediately.

   # (Optional manual setup) if you want to Cuztomize the Model setup.
   main_model.mlp_training_epochs = 2000
   main_model.transformer_training_epochs = 200
   main_model.transformer_lr = 0.25
   main_model.mlp_lr = 0.5
   main_model.lstm_training_epochs = 200
   main_model.lstm_lr = 5e-1
   main_model.lstm_hidden_dim = 64 # LSTM Hidden dim can be set manually, MLP and Transformer hidden dim are already set automatically based on the samples input they have received.
   # Transformer heads and d_model manual setup
   main_model.transformer_heads = 4
   main_model.transformer_d_model = 64
   # You can set how much epochs are needed to Train your MLP, LSTM and Transformer for your Models, along with their Learning rates. (lr).
   
   # test samples with more sophisticated rules and more complex titles for prediction
   # (title, intent)
   test_titles = [
    ("Opening Thesis.docx", "slight_work"),
    ("Watching YouTube and Google Chrome", "distracted"),
    ("Watching Slack", "communication"),
    ("Programming in Visual Studio Code", "focused_work"),
    ("Watching netflix.com - Chrome", "break"),
   # more titles 
    ]  
               
   titles, y, label_map = main_prediction.load_labels_from_csv(
      <your_filename.>,  # the name of your .txt file with CSV format.
      <target_title>, <target_label>)

   # OPTIONAL TRAINING BLOCK
   # (Training will happens in the advanced prediction method below, but if you want separate Training for MLP, you can use this setup: )
   # small training with simple titles first
   main_model.train(titles, y)

   # main_model.freeze_learning = True
   # prevent the model from training and make weights unchanged for static prediction.

   # SIMULATION BLOCK:
   # This function below allows you to create A continuous Training simulations using K-fold split to evaluate MLP Training performance using your provided X and Y samples, This function helps to diagnose if Your samples are appropriate for MLP and make It generalize instead of severely Overfitting.
   # This function will create a fresh instance of MLP every k-amount you have set (the k=5), and also prints a Confusion matrix that helps diagnose Your MLP Training Accuracy Performance.
   # This function is great to help you diagnose Further degradation of MLP performance and also Its consistency, And also helps validate our AWE Method at how well it makes MLP Training performance Consistent Accross different Seeds for better Reliability performance.
   main_model.evaluate_mlp_performance(X, y, label_map, k=5, seed=42)
   
   # PREDICTION BLOCK:
   # the below section called advanced_prediction_method(...) is a prediction method that will output a single answer of a problem you have given to it, it will output a single answer from the label_map you have given as its final prediction.
   # meaning advanced_prediction method is only used to predict an answer based on the given label_map and only output a single answer, not in batches.
   # this prediction method is also where training, ensemble and final prediction happens.
   # Use case: - classification problems that requires a model to only output a single answer.
   results, chosen_label, confidence = main_prediction.advanced_prediction_method(
      titles=test_titles, label_map=label_map, rules=example_rules, # titles and rules can be set to None (Optional samples), but label_map must NOT be None.
         X=None, y=None # you could create your own X and y samples and put it here (Optional, y sample must already be one hot encoded first).
               show_proba=False, top_k=3, 
               use_transformer=True,
               return_attention=False,
               batch_size=2)
   # Important Note: If you set titles and rules to None, you must provide X and y samples for prediction, otherwise the models cant predict anything.
   # Note: The X and y samples will be organized and processed using train_test_split() scikit-learn function for creating better generalization behavior for the model, so when you pass the X and y samples, you must pass the raw X sample (Not modified, just raw X) and the already y hot-encoded sample,
   # batch size=2 is needed during transformer training for batching, if you have larger samples consider using batch_size > 8, for medium amount of samples (>10 -> <50 samples) consider using 2 or 4 batch_size.

   # This setup below would allow you to save the Accurate answer (if the model guessed a specific problem correct) directly to the database,
   # without initiating prediction over the sample repeatedly.
   input_ids = main_model.model2.cache['input_ids'] # input indices the transformer used as input (model2 is Transformer).
   index = results.get('index') # predicted index where the Model have chosen the label.
   confidence = results.get('confidence') # the model confidence over the predicted Output from advanced prediction method.
   main_model.accurate_cache_lookup.add_verified(X, input_ids, chosen_label, confidence, index,
                     source='Correctly-Answered') # You can modify the the source with "Answered-Correctly" (fit to your needs)
   
   # ... more features you can add
   ```
   - Note: This script setup can be downloaded here: [usage_script](scripts/usage_scripts.py)
   
4. B. Using Standalone IntegratedPipeline Transformer:
   ```python
   # if you want to use the IntegratedPipeline Transformer only, you can use this setup:
   # ideal when you want blazing Fast Transformer Training and prediction, and the Transformer weights are loaded using JSON, this made Transformer Training much more feasible and easily intuitive for Beginners who wants Complex abstraction models that can run on cheap Edge devices.

   vocab_size = # <your actual vocab size>.
   model_transformer = Transformer(
                  vocab_size=vocab_size
                  d_model=main_model.transformer_d_model,
                  n_heads=pipeline.transformer_heads,
                  num_classes=num_classes
              )

   sequence_inputs = main_model._features_to_sequence(X, d_model=pipeline.transformer_d_model) # Converts X samples into sequences that the Transformer can recognize.
   # embedded=True will ensure that Transformer forward method will put a correct mask for the samples.
   # mode='dynamic_backward' helps Transformer to gains better accuracy for Complex vocabularies.
   model_transformer.train(sequence_inputs, y, epochs=100, mode='dynamic_backward', lr=0.1, embedded=True, batch_size=2)

   # save Transformer weights
   tf = model_transformer
   json_data = {
        'token_embedding': tf.token_embedding,
        'pos_embedding': tf.pos_embedding,
        'W_q': tf.W_q,
        'W_k': tf.W_k,
        'W_v': tf.W_v,
        'W_q_fixed': tf.W_q_fixed,
        'W_k_fixed': tf.W_k_fixed,
        'W_v_fixed': tf.W_v_fixed,
        'W_o': tf.W_o,
        'ffn1': tf.ffn1,
        'ffn2': tf.ffn2,
        'ln1_scale': tf.ln1_scale,
        'ln1_shift': tf.ln1_shift,
        'ln2_scale': tf.ln2_scale,
        'ln2_shift': tf.ln2_shift,
        'output': tf.output,
        'output_bias': tf.output_bias
    }
    serializable_data = {key: val.tolist() for key, val in json_data.items()}
    with open("transformer_weights.json", "w") as file:
        json.dump(serializable_data, file, indent=4)  # indent adds readable formatting

    #load Transformer weights
    # in here transformer_model a new IntegratedPipeline Transformer instance:
    with open('transformer_weights.json', 'r') as file:
        loaded_data = json.load(file)
    transformer_model.W_q = loaded_data['token_embedding']
    ... # and so on until output_bias.
   ```
   - Note: script setup can be downloaded here: [transformer_usage](scripts/transformer_usage.py)
   
5. C. Use kNN-Augmented Transformer:
   Note: kNN-Augmented Transformer doesn't have its own Prediction pipeline flow, meaning after the Transformer forward method is called, the probabilities of the kNN Transformer will be returned and no prediction is made, We decided to be better this way because it grants Flexibility over how Prediction will be made in your favour:
   - Example Code:
   - ```python
      # knn_forward_inference() lives inside PipelinePredictionManager, so you need to call this class to initiate the knn_forward_inference() function.
      losses, accs = main_prediction.knn_forward_inference(X, y, memory_metric='euclidean', training=True, batch_size=2, train_mode='dynamic_backward', lr=0.1) # for Training kNN Transformer only
      transformer_probs, attn_weights = main_prediction.knn_forward_inference(X, y, memory_metric='euclidean', training=False) # for returning kNN Transformer probabilities and attention weights only.
      # Note: - train_mode can be set to 'dynamic_backward' if you have very large dataset, this makes Transformer Q, K, V be much more dynamic and grants flexible learning behavior for large dataset.
              # - train_mode can be set to 'fixed_backward' if you have small dataset, this makes Transformer Q, K, V to stay frozen so the FFN flow will handle the Training, making Learning in very little samples possible and deterministic in behavior.
              # - y sample must be one-hot encoded manually before its passed to the function, since the function above will not automatically one-hot encode the y-sample.
     
      main_prediction.pipeline.storage.save_hnsw_memory(memory_name) # This function saves the HNSW setup inside KNNAugmentedTransformer class to the the same database (activity_log.db) so it can be retrieved as Retrievable Memory later.
      main_prediction.pipeline.storage.save_memory_head(memory_name) # This function saves the Per-Head memory setup inside KNNAugmentedTransformer class to the the same database (activity_log.db) so it can be retrieved as Retrievable Memory later.
     
      main_prediction.pipeline.storage.load_hnsw_setup(memory_name) # This function is used to load the saved HNSW Memory inside your (activity_log.db) database, and automatically apply the Old Memory back to the knn-augmented Transformer.
      main_prediction.pipeline.storage.load_head_memory_setup(memory_name) # This function is used to load the saved HNSW Memory inside your (activity_log.db) database, and automatically apply the Old Memory back to the knn-augmented Transformer.

     # .... # your own custom prediction block.
     ```
      
     
7. To use IntegratedPipeline prediction without Transformer, Only Specialized MLP:
      Note: IntegratedPipeline without Transformer is'nt recommended due to it being weak at certain contextual prediction's, excel's at classification task's.
      - Example without transformer:
   ```python
   prediction_result = main_prediction.advanced_prediction_method( 
            [t[0] for t in test_titles],  # titles is enough for MLP Classification.
            label_map,
            example_rules,
            show_proba=True
            )
   
   ```
8. Asynchronous prediction:
  - Asynchronous prediction request is important and is critical because it keeps prediction interfaces responsive, maximizes local hardware efficiency, and enables apps to handle background tasks seamlessly without waiting on remote server responses,
  - for asynchronous prediction handling, consider using this setup
```python
# Step 2
from AbstractIntegratedModule import PipelineAsyncManager
from AbstractIntegratedModule import SecurityConfig
from AbstractIntegratedModule import SecurityLevel

print(" = TESTING ASYNCHRONOUS PREDICTION MANAGER = ")
# Set discovery secret (in production, use environment variable)
secret_key = 'my-ultra-safe-secret-key-for-authentication' # you can customize this key


security_config = SecurityConfig(
      max_text_length=10000, # can be extended
      max_queue_size=100, # can be extended
      rate_limit_requests=60,  # 60 per minute
      require_api_key=True, #
      max_pending_tasks=50,
      request_timeout=60.0,

      # Start with no IP restrictions, you can add allowed IPs for asynchronous prediction externally, boothstrap_auth for better security
      allowed_ips=[],
      blocklisted_ips=[],
      require_bootstrap_auth = False # true for better security (Not recommended, cause less flexibility)
  )

async_manager = PipelineAsyncManager(main_model, 
        main_prediction, # your previous initialized PipelinePredictionManager
        config=security_config, 
        state_file=None, # state file is used to load known security logs ex: ip used, ip blacklisted, etc.
        security_level=SecurityLevel.PRODUCTION, # production level security initiated
        api_key=secret_key  #set secret key you initialized
        max_workers=4, # workers to initiate asynchronous tasks, more workers, more capabilities to process asynchronous prediction requests.
        task_timeout=30, 
        max_retries=3 ) # retries after failure during prediction

async_manager.start(method='Transformer_included', bootstrap_token=None) # boothstrap token is optional for better security

texts = {'test_titles': test_titles, 'label_map': label_map, 'rules': example_rules, # test_titles and rules can be set to None here since they are Optional samples.
                                                                                     # but label_map must not be None.
          'X': None, 'y':None, 'use_transformer': True} # all samples needed for advanced prediction method. (X and y are optional samples)
# Important Note: If you set titles and rules to None, you must provide X and y samples for prediction, otherwise the models cant predict anything.

regular_predict = async_manager.predict(
   texts=texts,
   timeout=60,
   retries=None,
   api_key=secret_key) # advanced prediction method for asynchronous prediction.

# with retries: async_manager.predict(texts, timeout=60, retries=5, api_key=secret_key) # 5 times retry if failed

# NOTE: This function below must require test_titles or titles, label_map and example rules:
print('[==] Initiating advanced batch prediction')
         predicted_output = async_manager.advanced_batch_prediction(test_titles, label_map, example_rules, 
         X=None, y=None, # provide your initialized X and y samples (Also Optional, can be set to None)
         secret_key=secret_key, client_ip=None) # you can add client_ip to provide a robust authentication paired with secret_key
# for better and faster advanced prediction when using titles and rules, consider using advanced batch prediction like in the above example

```
[=] Note:
 - Asynchronous prediction used Event loop that handles incoming request, There are conditions where event loop will not start and can't accet requests:
   - CPU Above > 95%    - Disk space is < 100 MB
   - RAM above > 95%
 - When event loop is not triggered, Asynchronous prediction can't be initiated and must be restarted/retried.
 - Script setup can be downloaded here: [async_script](scripts/async_script.py)

8. Peer-to-Peer Probability coordination:
   - Each peer is both server and client simultaneously for robustness and resilience during during P2P.
   - To Make the Agent cooperate with other peers, consider using this setup:
   - [=] for ensemble prediction from multiple peers, exchanging predicted label with each other, consider using this setup:
```python

# step 3
from AbstractIntegratedModule import CohesiveAgentDeployment
from AbstractIntegratedModule import PipelinePredictionManager
import asyncio
import traceback

prediction_manager = PipelinePredictionManager(main_model, label_csv=<your_training_labels.txt>, target_title=<target_title>, label=<target_label>)

secondary_model = IntegratedPipeline(memory_name=memory_name, 
                use_async=True, agent_port=8080, 
                ssl_cert_file=cert_file, ssl_key_file=key_file) # provide cert_file path or key_file path (optional)
# secondary model of integrated pipeline is critical for ARM64 environment to prevent socket conflict during P2P with the first Integrated pipeline instance.
# make sure that the agent_port in secondary_model must be different from agent_port in first IntegratedPipeline instance you initialized.

print("=== SECURE PEER-TO-PEER CLUSTER ===")

# CohesiveAgentDeployment is deeply tied and coupled with AgentDistributedInference,
# if you already set an SSL cert and key, CohesiveAgentDeployment will use the SSL directly from AgentDistributedInference
# allowing secure socket to be used directly by CohesiveAgentDeployment

main_model.distribution.enable_ssl = False # set to false if you dont have SSL key and CERT, this code would instruct AgentDistributedInference that you don't have SSL, and provide you a regular unsecured socket (Not necessary for production)
secondary_model.distribution.enable_ssl = False

# Agent 1 - Primary (Port 5555)
agent1 = CohesiveAgentDeployment(
     pipeline=main_model, # main_model is your initialized integrated pipeline
     memory_name="agent_primary", # any name you want for the agent.
     filename=<filename>, # name of your .txt file that contains the CSV format and training labels
     target_title=<title_name>, 
     label_name=<label_name>,
     security_level="PRODUCTION", # production level security
     enable_peers=True, # allow peer discovery
     trusted_networks=['127.0.0.1/32', '192.168.1.0/24'], # for trusted networks, you need to provide the list of IPs of your peers.
     peer_discovery_port=5555, # peer port to start P2P
     secret_key=secret_key, # your secret key
     shared_auth_token=secret_key, # your previous initialized secret_key
     predict_manager=prediction_manager, # your prediction manager
     peer_config = <'your_peer_ip_lists.json'> # you need to create .json file that contains your peer IP and Port lists
     consecutive_peer_config = <'your_second_fallback_peer_ip_lists.json'> # same for this one too, but for fallback.
     )
 
# Agent 2 - Secondary (Port 5556)
agent2 = CohesiveAgentDeployment(
     pipeline=secondary_model,
     memory_name="agent_secondary",
     filename=<filename>,
     target_title=<title_name>,
     label_name=<label_name>,
     security_level="PRODUCTION",
     enable_peers=True, # agent is allowed to find peers
     trusted_networks=['127.0.0.1/32', '192.168.1.0/24'],
     peer_discovery_port=5556,
     secret_key=secret_key,
     shared_auth_token=secret_key,
     predict_manager=prediction_manager,
     peer_config = <'your_peer_ip_lists.json'> # you need to create .json file that contains your peer IP and Port lists
     consecutive_peer_config = <'your_second_fallback_peer_ip_lists.json'> # same for this one too, but for fallback.
     )

# Note: CohesiveAgentDeployment contains ConsecutivePeerAgent that can start a server once ensemble prediction from peer is started
# be advised to stop the server too before shutdown-ing CohesiveAgentDeployment cluster

# example peer_Ip_lists_config.json (de-comment to use)
# {
      # you must put "known_peers" in the config so python can identify the list of IPs and Ports 
      #  "known_peers": [ 
        #    ["127.0.0.1", 5555], can be modified using real IP or local IP.
        #    ["127.0.0.1", 5556],
             # more ip and port lists...
     #   ]
    # }


try:
     # Start both agents
     print("\n🚀 Starting Agent 1...")
     await agent1.start()
     print("✅ Agent 1 started on port 5555")
     
     print("\n🚀 Starting Agent 2...")
     await agent2.start()
     print("✅ Agent 2 started on port 5556")
     
     # Give servers time to fully bind
     await asyncio.sleep(2)
     
     # Get API keys
     api_key = agent1.get_api_key()
     print(f"\n🔑 Using API Key: {api_key[:20]}...")

     texts = {"test_titles": test_titles, "label_map": label_map, "rules": rules, # test_titles and rules are Optional samples, can be set to None.
'X': None, "y":None, "use_transformer": True, "agent_id": agent_id} # (X and y are optional samples here too)
     # Important Note: If you set titles and rules to None, you must provide X and y samples for prediction, otherwise the models cant predict anything.

     # texts dictionary must contain test_titles, label_map, and rules that you can assign,
     # agent ID can be strings, int, or floats, recommendded to make it long for better security.
     agent1.pipeline.use_transformer = False # set agent1 pipeline to not use transformer for efficient processing in ARM64 env.
     agent2.pipeline.use_transformer = False
      
      # Make prediction with peer ensemble
      # Connection will be guaranteed successfull during discovery.
     result = await agent1.multi_modal_peer_ensemble_prediction(
          texts=texts,
          api_key=api_key,
          method='advanced',
          disable_sync=True
      )  # await using asyncio, multi_modal_peer_ensemble is already async by design (Inside ConsecutivePeerAgent), no need to put asyncio.run()

     # must have agent2 peer ensemble function after agent1 peer ensemble prediction function, 
     # so agent1 can receive agent2 peer ensemble prediction request.
     result2 = await agent2.multi_modal_peer_ensemble_prediction(
          texts=texts,
          api_key=api_key,
          method='advanced',
          disable_sync=True
      )

     print(f"\n📊 Ensemble Result for Agent 1:")
     print(f"   Prediction: {result.get('prediction', 'N/A')}")
     print(f"   Confidence: {result.get('confidence', 0):.2%}")

     print(f"\n📊 Ensemble Result for Agent 2:")
     print(f"   Prediction: {result2.get('prediction', 'N/A')}")
     print(f"   Confidence: {result2.get('confidence', 0):.2%}")

     # Keep running briefly
     print("\n⏳ Cluster stable. Waiting 5 seconds before shutdown...") # 5 seconds before shutdown.

     # stop ConsecutivePeerAgent servers inside CohesiveAgentDeployment.
     agent1._peer_agent.stop_server() # ._peer_agent is ConsecutivePeerAgent
     agent2._peer_agent.stop_server()

     await asyncio.sleep(5)

except Exception as e:
     print(f"\n❌ Error in cluster: {e}")
     traceback.print_exc()
     
finally:
     print("\n🛑 Shutting down cluster...")
     await agent1.shutdown()
     await agent2.shutdown()
     print("✅ Cluster shutdown complete")

```
[=] Important Note: 
    - This setup outputs the given predicted label of an input directly, making P2P more flexible and fast.
    - This setup used Hybrid feature in prediction handling, Asynchronous prediction request, and Synchronous prediction handling. Synchronous prediction does block code execution for a few seconds, it was used for a few reason here, such as:
       - allowing a more slower traffic between agents, preventing other agent to get the same peer prediction over time, making each interaction equals and each peer can receive different peer prediction output.
       
[=] You can download this setup here for a direct test: [P2PDirectTest.py](P2P_Setups/P2PDirectTest.py)

   - [=] for probability coordination, locally, get peers data from database or via socket.
```python
# step 4
dataset, _ = main_model.data_preparation(test_titles, label_map)
sequence_inputs = main_model.sequence_encoding(dataset)
X_raw_generation, y, n_classes, input_dim = main_model.mlp_training_features(example_rules, dataset)

main_model.initialize_fitting(dataset)
X_raw_features = main_model.tfidf.transform(X_raw_generation).toarray()
transformer_features = main_model.transformer_pooled_features(sequence_inputs)
X_features = np.concatenate([X_raw_features, transformer_features], axis=-1)

peer_probability_calibration = main_model.predict_proba(sequence_inputs, X_features, type='Hybrid', embedded=True) # peer-to-peer calibration is inside this function
```
[~] Note: the peer calibration coordination has a chance of triggering if both MLP and Transformer prediction doesn't agree on certain output. Consider using this setup below for using stand-alone peer-to-peer main function without being wrapped in other parent function, allowing flexible and auditable peer-to-peer sharing for probability coordination:
```python
from AbstractIntegratedModule import WeightedEnsemblePredictor
from AbstractIntegratedModule import Transformer

num_classes = len(label_map)
# if you haven't fit the Tfidf:
# main_model.initialize_fitting(dataset)

ensemble_method = WeightedEnsemblePredictor(main_model, memory_name) # consider using the same memory name used in your previous pipeline
transformer = Transformer(main_model.vocab_size, d_model=32, n_heads=4, num_classes=num_classes) # you can audit how much parameter the transformer needs.
main_model.model2 = transformer # overwrite previous transformer initialization

# main_model.distribution is AgentDistributedInference() class
# consider using ssl for secure peer to peer coordination
main_model.distribution.ssl_cert_file = <path_to_your_ssl_cert_file> 
main_model.distribution.ssl_key_file = <path_to_your_ssl_key_file>

dataset, _ = main_model.data_preparation(titles, label_map)
sequence_input = main_model.sequence_encoding(dataset)
_, attn_weights = transformer.forward(sequence_input)

probs = ensemble_method.predict_ensemble(sequence_input, X_features, y, method='dynamic', embedded=True)
# 3 options for ensemble weighting method:
# 1. dynamic: allow flexible, efficient weighting from both transformer and MLP,
# 2. meta: for a much more in-depth weighting from both model,
# 3. calibration: allow calibrating probability for both model outputs based on both best weights assembling.

agreement = main_model.agreement

# start server to initiate socket for P2P listener
main_model.distribution.start_server()

# set connection timeout (Optional)
main_model.distribution.connection_timeout = 30 # 30 seconds before timeout
calibrated_probability = main_model._handle_distributed_connections(probs, attn_weights, sequence_input, agreement)

# if an Agent experience a failure on tasks, consider using this function to reduce peer trust for safer flexible coordination:
# main_model.distribution.report_failure(id(main_model), '<task_name>', reason='<unknown>') # you can add the task_name and reason
# main_model.distribution.print_network_status() # to show other peers info.
```
[~] Note: this calibrated_probability is later used to calculate confidence and chosen output based on given label_map.
   - Consider checking:
     - [multi_agent_client.py](P2P_Setups/multi_agent_client.py) for a In-depth start for client testing.
     - [multi_agent_server.py](P2P_Setups/multi_agent_server.py) for a In-depth start for server testing.
   - If you get undefined NoneType Behavior when using .accept(), consider see [Troubleshooting](#Troubleshooting) Issue 7 for a Quick fix.
        
6. Cross-Session availability:
   - To use Cross-session avialability to transfer or import memory, consider using this setup:
     - ```python
       main_model._cross_session_availability() # cross session capability function
       ```

7. As an option, You can add more feature's directly to what it should predict, behave using rules you have given, Create a visual dashboard, create a distributed mesh of this agent, and much more features you can try.
