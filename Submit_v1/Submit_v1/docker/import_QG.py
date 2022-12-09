import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
logging.getLogger("transformers").setLevel(logging.ERROR)

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
qa_model = pipeline("question-answering")