#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install goslate')


# In[3]:


get_ipython().system('pip install translate')


# In[5]:


from translate import Translator
translator = Translator(to_lang="ta")
translation = translator.translate("How are you?")
print(translation)


# In[7]:


from translate import Translator
translator = Translator(to_lang="hi")
translation = translator.translate("What is your name?")
print(translation)


# In[8]:


get_ipython().system('pip install python-vlc')


# In[12]:


import vlc
p = vlcMediaPlayer("audio.mp3")
p.play()


# In[10]:


get_ipython().system('pip install playsound')


# In[13]:


import playsound as pl
pl.playsound('audio.mp3')
print('playing sound using playground')


# In[ ]:




