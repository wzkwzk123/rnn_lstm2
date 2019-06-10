#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn


# In[ ]:


class Dictionary(object):
    '''
    构建word2id,id2word两个字典
    '''
    def __init__(self):
        self.word2idx = {} #dictionary word:index
        self.idx2word = {} #dictionary index:word
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx: #  not add the same word which is already in 
            self.word2idx[word] = self.idx 
            self.idx2word[self.idx] = word #Create an index-to-word mapping
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx) #Dictionary length 


# In[2]:


class Corpus(object):
    '''
    Build dictionaries word2id,id2word based on Corpus (training data)
    '''
    def __init__(self):
        self.dictionary = Dictionary() # an instance of the Dictionary Class

    def get_data(self, path, batch_size=20,seq_length = 25):
        batch_size = batch_size
        seq_length = seq_length
        with open(path, 'r') as f:# Read files line by line
            tokens = 0
            for line in f:  
                words = line.split() + ['<eos>'] 
                tokens += len(words) #The total number of words in the file  
                for word in words: 
                    self.dictionary.add_word(word)  
        ids = torch.LongTensor(tokens)
        
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word] 
                    token += 1
                num_batches = ids.size(0) // batch_size
                num_seq_eachline = num_batches // seq_length
        ids = ids[:seq_length*num_seq_eachline*batch_size]

        return ids.view(batch_size, -1) 

