#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers torch')


# In[2]:


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_name = "cardiffnlp/tweet-topic-21-multi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
labels = [
    "arts&_culture", "business&_entrepreneurs", "celebrity&_pop_culture", 
    "diaries&_daily_life", "family", "fashion&_style", 
    "film_tv&_video", "fitness&_health", "food&_dining", 
    "gaming", "learning&_educational", "music", 
    "news&_social_concern", "other&_hobbies", "relationships", 
    "science&_technology", "sports&_esports", "travel&_adventure", 
    "youth&_student_life"
]


# In[3]:


texts = [
    "The latest iPhone was just released with an incredible new camera!",
    "Manchester United won their match with a stunning goal in the last minute.",
    "NASA just launched a new mission to explore the surface of Mars.",
    "The Oscars had some surprising winners this year!"
]


# In[5]:


inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
predictions = torch.argmax(probabilities, dim=1)
for text, pred, prob in zip(texts, predictions, probabilities):
    print(f"Text: {text}\nTopic: {labels[pred.item()]}, Confidence: {prob[pred].item():.4f}\n")


# In[6]:


from transformers import pipeline
summarizer = pipeline("summarization")
text = """Hugging Face is a company that specializes in natural language processing (NLP).
It has developed the Transformers library, which provides state-of-the-art models
for a wide range of NLP tasks such as text classification, information extraction,
question answering, summarization, translation, and more. The library is widely used
in both academia and industry due to its ease of use and flexibility."""


# In[9]:


summary=summarizer(text, max_length=50, min_length=20, do_sample=False)
print("Summary:", summary[0]['summary_text'])


# In[10]:


from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
prompt = "Once upon a time in a distant galaxy,"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_length=50, num_return_sequences=1, temperature=0.7, top_k=50)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)


# In[ ]:


import os
import atexit
import shutil
from transformers import Blenderbot

