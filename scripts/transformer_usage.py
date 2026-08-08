# if you want to use the IntegratedPipeline Transformer only, you can use this setup:
# hybrid method, combining Pytorch image processing with IntegratedPipeline Transformer properly:
# ideal when you want blazing Fast Transformer Training and prediction, and the Transformer weights are loaded using JSON, this made Transformer Training much more feasible and easily intuitive for Beginners who wants Complex abstraction models that can run on cheap Edge devices.
import torch
...

loader = get_custom_dataloaders(...)
num_classes = #<your_data_classes>
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_transformer = Transformer(
               vocab_size=1,  # set to always 1 because the Transformer is used to analyze images, not Texts.
               d_model=main_model.transformer_d_model,
               n_heads=pipeline.transformer_heads,
               num_classes=num_classes
           )
for x, y in loader: # extract proper X and Y samples.
   x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

x = x.flatten(start_dim=1).detach().cpu().numpy() # flatten X to be compatible for later pipeline processing, and convert X into standalone numpy object.
y = y.detach().cpu().numpy() 
y = np.eye(num_classes)[np.asarray(y)] # proper y-onehot

sequence_inputs = main_model._features_to_sequence(x, d_model=pipeline.transformer_d_model) # Converts X samples into sequences that the Transformer can recognize.
# embedded=True will ensure that Transformer forward method will put a correct mask for the samples.
# mode='dynamic_backward' helps Transformer to gains better accuracy for Complex images.
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
