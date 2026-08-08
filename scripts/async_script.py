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
