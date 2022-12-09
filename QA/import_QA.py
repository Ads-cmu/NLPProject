from haystack.nodes import FARMReader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

logging.disable(logging.INFO)
logging.disable(logging.WARNING)


reader = FARMReader(model_name_or_path="deepset/tinyroberta-squad2", use_gpu=False, progress_bar=False)

qa_tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
bool_qa_model = AutoModelForSequenceClassification.from_pretrained("/bool_qa_model/")

summarizer = pipeline("summarization", model="facebook/bart-base")
