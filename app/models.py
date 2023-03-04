from sentence_transformers import SentenceTransformer
import pickle
import transformers
import math

big_model = transformers.AutoModelForSequenceClassification.from_pretrained('app/models_saved/bert_final_model', local_files_only=True)
tokenizer_big_model = transformers.AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

with open("app/models_saved/log_reg_emb.pkl", "rb") as file:
    small_model = pickle.load(file)

small_model_emb = SentenceTransformer('DeepPavlov/rubert-base-cased', device="cpu")


def log_reg_model(processed_text):
    emb = small_model_emb.encode([processed_text])
    return float(small_model.predict_proba(emb)[0][1])


def bert_model(processed_text):
    logits = big_model(
        **tokenizer_big_model(processed_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        ).logits.item()
    return 1 / (1 + math.exp(-logits))
