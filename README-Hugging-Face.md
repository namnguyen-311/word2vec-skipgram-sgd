# Hugging Face Q&A Chatbot

An interactive question-answering bot built with Hugging Face Transformers.  
It takes a custom context and answers user questions using pretrained NLP models.

## Features
- Powered by Hugging Face's `pipeline("question-answering")`
- Runs in terminal with real-time user input
- Uses pretrained models (BERT-style) under the hood
- Example context: Word2Vec explanation from CS224N

## Example
# You: What is Word2Vec used for?
# Bot: to learn vector representations of words

## Requirements
- Python 3.7+
- `transformers`
- `torch`

## Run It
```bash
pip install torch transformers
python qa_chatbot.py
