from AbstractIntegratedModule import IntegratedPipeline
from AbstractIntegratedModule import PipelinePredictionManager
import numpy as np

memory_name = 'agent_memory'
main_model = IntegratedPipeline(memory_name=memory_name, use_async=True, agent_port=8080,ssl_cert_file=None, ssl_key_file=None) # provide cert_file path or key_file path (optional)
main_prediction = PipelinePredictionManager(main_model, 
                      label_csv='C:/your/path/to/ManualsTraining.txt', # you could change this with your actual directory.', 
                      target_title='window_title', 
                      label='label')
# example_manual_training is a .txt file that contain csv format like above example.

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
main_model.show_explainability_details = False
main_model.distribution.predict_manager = main_prediction

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
    'C:/your/path/to/ManualsTraining.txt', # you could change this with your actual directory.
    'window_title', 'label')

# small training with simple titles
main_model.train(titles, y)
   
results, chosen_label, confidence = main_prediction.advanced_prediction_method(test_titles, label_map, example_rules,
                             X=None, y=None, # add optional samples
                             show_proba=False, top_k=3, 
                             use_transformer=True,
                             return_attention=False,
                             save_results=True,
                             batch_size=2)



dataset, _ = main_model.data_preparation(titles, label_map)
sequence_inputs = main_model.sequence_encoding(dataset)
X_raw_generation, y, n_classes, input_dim = main_model.mlp_training_features(example_rules, dataset)

main_model.initialize_fitting(dataset)
X_raw_features = main_model.tfidf.transform(X_raw_generation).toarray()
transformer_features = main_model.transformer_pooled_features(sequence_inputs)
X_features = np.concatenate([X_raw_features, transformer_features], axis=-1)

peer_probability_calibration = main_model.predict_proba(sequence_inputs, X_features, type='Hybrid', embedded=True) # peer-to-peer calibration is inside this function

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
        api_key=secret_key, #set secret key you initialized        
        max_workers=4, # workers to initiate prediction, more workers, more capabilities to process prediction requests.
        task_timeout=30, 
        max_retries=3 ) # retries after failure during prediction

async_manager.start(method='Transformer_included', bootstrap_token=None) # boothstrap token is optional for better security

texts = {'test_titles': test_titles, 'label_map': label_map, 'rules': example_rules, 'use_transformer': True}
regular_predict = async_manager.predict(
   texts=texts,
   timeout=60,
   retries=None,
   api_key=secret_key) # advanced prediction method for asynchronous prediction.

# with retries: async_manager.predict(texts, timeout=60, retries=5, api_key=secret_key) # 5 times retry if failed

print('[==] Initiating advanced batch prediction')
predicted_output = async_manager.advanced_batch_prediction(test_titles, label_map, example_rules, secret_key, client_ip=None)
# for better and faster advanced prediction, consider using advanced batch prediction like in the above example


# ... more features you can add
