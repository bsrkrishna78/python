#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nltk')


# In[2]:


import nltk
import re
from nltk.chat.util import Chat, reflections


# In[3]:


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# In[4]:


pairs = [
    [r"my name is (.*)", ["Hello %1, how can I assist you today?"]],
    [r"hi|hey|hello", ["Hello, how can I help you?", "Hey there! What can I do for you?", 
                       "Hi! How can I assist you today?"]],
    [r"what is your name?", ["I am a chatbot created to assist you. You can call me Chatbot."]],
    [r"how are you?", ["I'm a bot, so I don't have feelings, but I'm here to help you!"]],
    [r"can you help me with (.*)", ["Sure, I can help you with %1. Please provide more details."]],
    [r"sorry (.*)", ["It's okay. How can I assist you?"]],
    [r"thank you|thanks", ["You're welcome!", "No problem!", "Happy to help!"]],
    [r"quit", ["Bye! Have a great day!", "Goodbye!"]],
    [r"(.*)", ["I'm sorry, I don't understand that. Can you rephrase?", 
               "Could you please elaborate on that?"]]
]


# In[5]:


class RBChatbot:
    def __init__(self, pairs):
        self.chat = Chat(pairs, reflections)
    def respond(self, user_input):
        return self.chat.respond(user_input)
chatbot = RBChatbot(pairs)


# In[7]:


def chat_with_bot():
    print("Hi, I'm your chatbot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower()  == 'quit' :
            print("Chatbot: Bye! Have a great day!")
            break
        response = chatbot.respond(user_input)
        print(f"Chatbot: {response}")
        
chat_with_bot()


# In[ ]:




