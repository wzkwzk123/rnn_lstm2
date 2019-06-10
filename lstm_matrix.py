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
        self.W_x_h = nn.Parameter(torch.Tensor(4*self.hidden_size, self.embed_size)) 
        self.b_x_4 = nn.Parameter(torch.Tensor(4*self.hidden_size)) 
        self.W_h_h = nn.Parameter(torch.Tensor(4*self.hidden_size, self.hidden_size)) 
        
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
        
        # [24, 20,128]  * [1024*4,128]' +[1024*4]==>[24,20,1024*4]+[1024*4] = [24,20,1024*4]      
        x_4 = F.linear(x, self.W_x_h, self.b_x_4)   


        h_4 = F.linear(h_pre, self.W_h_h)       #[20, 1024]  *[1024*4,1024]' ==>[20,1024*4]  这24 个单词都是公用的 w和 b
        xh_4 = x_4 + h_4                  #[24,20,1024*4]  +   [20,1024*4]  =  [24,20,1024*4] 
        xh_i, xh_f, xh_o, xh_c = xh_4.chunk(4, 2)   #[24,20,1024*1]  在维度2上拆成4份
        

        i_gate = torch.sigmoid(xh_i)
        f_gate = torch.sigmoid(xh_f)
        o_gate = torch.sigmoid(xh_o)   
        c_hat = torch.tanh(xh_c)

        #这种写法是dot multi!!!
        c = f_gate * c_pre + i_gate * c_hat   #[24, 20, 1024] .*[20, 1024]  + [24, 20, 1024]*[20,1024]=[24,20,1024] 
        out = o_gate * torch.tanh(c)       #[24,20,1024] .*[24,20,1024]


        
        ## as h_pre and c_pre for the next batch
        h_last = out[-1]  #out[out.size(0)-1]   [23,:,:]
        c_last = c[-1]  #c[c.size(0)-1]

        # fc layer :for loss caculation in this batch 
        out = out.permute(1,0,2) ##  because of this line the result is much better
        out = out.reshape(out.size(0)*out.size(1),out.size(2))  #[24,20,1024]=>[24,*20,1024]
        #[24*20,1024]=>[24,20*vocab_size]
        out = self.linear(out)  
        
        return out,(h_last,c_last)

