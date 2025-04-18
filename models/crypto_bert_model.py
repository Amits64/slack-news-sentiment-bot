from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

def load_crypto_bert_pipeline():
    model_name = "ElKulako/cryptobert"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    return TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_length=64,
        truncation=True,
        padding='max_length',
        return_all_scores=False  # important: 'False' for returning top labels only
    )
