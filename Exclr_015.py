#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install gTTs')


# In[4]:


from gtts import gTTS


# In[5]:


convert=gTTS(text='I like this NLP. How about dude!', lang="en", slow=False)
convert.save("audio.mp3")


# In[7]:


get_ipython().system('pip install pyttsx3')


# In[8]:


import pyttsx3
engine=pyttsx3.init()
engine.say("Hi, I am text to speach")
engine.runAndWait()


# In[9]:


text = ['This is introduction to NLP','It is likely to be useful, to people',\
        'Machine learning is the new electricity','There would be less hype around AI and more action going forward',\
        'python is the best tool!','R is good language',\
        'I want more books like this']


# In[10]:


engine=pyttsx3.init()
engine.say(text)
engine.runAndWait()


# In[11]:


import pyttsx3
engine = pyttsx3.init() 
""" RATE """
rate = engine.getProperty('rate')
print (rate)                       
engine.setProperty('rate', 150)   
""" VOLUME """
volume = engine.getProperty('volume')  
print (volume)                         
engine.setProperty('volume',1.0)      
""" VOICE """
voices = engine.getProperty('voices')        
engine.setProperty('voice', voices[1].id) 
engine.say("Hello World!")
engine.say('My current speaking rate is ' + str(rate))
engine.say('My current speaking volume is ' + str(volume))
engine.runAndWait()


# In[12]:


get_ipython().system('pip install goslate')


# In[ ]:




