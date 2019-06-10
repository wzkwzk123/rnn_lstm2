#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# In[5]:


# lstm language model
class Lstm(nn.Module):  
    def __init__(self, batch_size,vocab_size, embed_size, hidden_size,num_layers):
        super(Lstm, self).__init__()
        self.embed_size = embed_size        
        self.hidden_size = hidden_size  
        self.batch_size = batch_size

        #LSTM
      
        self.W_x_i = nn.Parameter(torch.Tensor(self.hidden_size, self.embed_size)) 
        self.W_x_f = nn.Parameter(torch.Tensor(self.hidden_size, self.embed_size)) 
        self.W_x_o = nn.Parameter(torch.Tensor(self.hidden_size, self.embed_size)) 
        self.W_x_c = nn.Parameter(torch.Tensor(self.hidden_size, self.embed_size))  
        
        self.b_x_i = nn.Parameter(torch.Tensor(self.hidden_size)) 
        self.b_x_f = nn.Parameter(torch.Tensor(self.hidden_size)) 
        self.b_x_o = nn.Parameter(torch.Tensor(self.hidden_size))   
        self.b_x_c = nn.Parameter(torch.Tensor(self.hidden_size)) 
        

        self.W_h_i = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size)) 
        self.W_h_f = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size)) 
        self.W_h_o = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size)) 
        self.W_h_c = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size)) 
        
        """
        only use one bias to decrease the number of weights
        h_f = F.linear(h_pre, self.W_h_i,self.b_h_i).....here we don't consider the b_h_i
               
        """
        #one-hot(vocab_size,1) -> (embed_size,1)
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        #fully connected layer  
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    
    def init_weights(self):
        init_range = 1. / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-init_range, init_range)     
        self.embed.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.uniform_(-init_range, init_range)
        self.linear.weight.data.uniform_(-init_range, init_range)


    def forward(self, x,output_pre,memory_pre):
        # Word embedding
        h_pre = output_pre  #[20, 1024]
        c_pre = memory_pre  #[20, 1024]
        x = self.embed(x)  #[24, 20] =>  [24, 20,128]   embed  前 是[24, 20] 的一个维度，但是有25689个可能性的词

        #x = x.view(x.size(0)*x.size(1),-1)      #[batch_size*(seq_length-1),embed_size]:[20,24,128]=>[20*24,128]

        x_i = F.linear(x, self.W_x_i, self.b_x_i)   # [24, 20,128]  * [1024,128]' ==>[24,20,1024]
        h_i = F.linear(h_pre, self.W_h_i)       #[20, 1024]  *[1024,1024]' ==>[20,1024]  这24 个单词都是公用的 w和 b
        
        #[24, 20, 1024] + [20, 1024]  = [24, 20, 1024]
        i_gate = torch.sigmoid(x_i+h_i)
        

        x_f = F.linear(x, self.W_x_f, self.b_x_f)  
        h_f = F.linear(h_pre, self.W_h_f)         
        
        #[24, 20, 1024] + [20, 1024]  = [24, 20, 1024]
        f_gate = torch.sigmoid(x_f+h_f)
        
        
        
        x_o = F.linear(x, self.W_x_o, self.b_x_o)   # [24, 20,128]  * [1024,128]' ==>[24,20,1024]
        h_o = F.linear(h_pre, self.W_h_o)       #[20, 1024]  *[1024,1024]' ==>[20,1024] 
        o_gate = torch.sigmoid(x_o+h_o)   

        x_c = F.linear(x, self.W_x_c, self.b_x_c)   # [24, 20,128]  * [1024,128]' ==>[24,20,1024]
        h_c = F.linear(h_pre, self.W_h_c)       #[20, 1024]  *[1024,1024]' ==>[20,1024] 
        c_hat = torch.tanh(x_c+h_c)
        
        c = f_gate * c_pre + i_gate * c_hat   #[24, 20, 1024] .*[20, 1024]  + [24, 20, 1024]*[20,1024]=[24,20,1024] 这种写法是dot mul!!!
        out = o_gate * torch.tanh(c)   #[24,20,1024] .*[24,20,1024]
        
        
        ## as h_pre and c_pre for the next batch
        h_last = out[-1]  #out[out.size(0)-1]   [23,:,:]
        c_last = c[-1]  #c[c.size(0)-1]

        # fc layer :for loss caculation in this batch 
        #[24,20,1024]=>[24*20,vocab_size]
        out = permute(1,0,2)  #[24,20,1024]=>[20*24,1024]  because of this line of code, the result is much better
        out = out.reshape(out.size(0)*out.size(1),out.size(2))  #[24,20,1024]=>[24,*20,1024]=
        out = self.linear(out)  
        
        return out,(h_last,c_last)

