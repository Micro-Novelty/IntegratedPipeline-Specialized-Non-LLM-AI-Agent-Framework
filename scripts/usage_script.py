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

# main_model.froze_learning = True
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
            save_results=True,
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
