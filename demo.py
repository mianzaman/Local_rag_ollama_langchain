from transformers import AutoModel, AutoTokenizer

model_name = "nlpaueb/legal-bert-base-uncased"

# Download and save locally
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer to a directory
model.save_pretrained("local_models/nlpaueb_legal_bert")
tokenizer.save_pretrained("local_models/nlpaueb_legal_bert")
