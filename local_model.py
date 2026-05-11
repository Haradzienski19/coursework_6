from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"

print("Скачивание модели...")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model.save_pretrained("./local_model")
tokenizer.save_pretrained("./local_model")

print("Модель сохранена в ./local_model")