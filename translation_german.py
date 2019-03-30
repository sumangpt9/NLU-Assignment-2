#!/usr/bin/env python
# coding: utf-8

# In[65]:




from __future__ import unicode_literals, print_function, division
from io import open
from numba import jit
import unicodedata
import string
import re
import random
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 100
SOS_token = 0
EOS_token = 1
batch_size=1
attn="multiplicative"


# 

# In[66]:


#torch.cuda.get_device_name(0)


# In[67]:


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s=s.replace(".","")
    s=s.replace("!","")
    s=s.replace("?","")
    #s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# In[68]:


#C_eng=pd.read_csv("commoncrawl.de-en.en",header=None)
#C_ger=pd.read_csv("commoncrawl.de-en.de",header=None)
#/commonCrawl
text_file = open("data/commoncrawl.de-en.en", "r")
C_eng = text_file.readlines()

text_file = open("data/commoncrawl.de-en.de", "r")
C_ger = text_file.readlines()

n=len(C_eng)


# In[69]:


pairs=list()
for i in range(40000):
    pairs.append([normalizeString(C_eng[i]),normalizeString(C_ger[i])])


# In[70]:


#print(len(pairs))


# In[71]:


tokenized_corpus_eng=[]
tokenized_corpus_ger=[]

for i in range(len(pairs)):
    
    tokenized_corpus_eng.append(pairs[i][0])
    tokenized_corpus_ger.append(pairs[i][1])


# In[72]:


vocabulary_eng = ["SOS","EOS","UNK"]
tokens_eng=[]
for sentence in tokenized_corpus_eng:
    

    for token in sentence.split(" "):

        tokens_eng.append(token)

        if token not in vocabulary_eng:
            vocabulary_eng.append(token)




word2idx_eng = {w: idx for (idx, w) in enumerate(vocabulary_eng)}
idx2word_eng = {idx: w for (idx, w) in enumerate(vocabulary_eng)}

vocabulary_size_eng = len(vocabulary_eng)


# In[73]:


vocabulary_ger = ["SOS","EOS","UNK"]
tokens_ger=[]
for sentence in tokenized_corpus_ger:
    
    
    for token in sentence.split(" "):
        tokens_ger.append(token)
        #if token.isnumeric():
            #tokens.append("num")
        if token not in vocabulary_ger:
            vocabulary_ger.append(token)

#print(vocabulary)

#print(vocabulary[0:10])


word2idx_ger = {w: idx for (idx, w) in enumerate(vocabulary_ger)}
idx2word_ger = {idx: w for (idx, w) in enumerate(vocabulary_ger)}

vocabulary_size_ger = len(vocabulary_ger)


# In[74]:



print(vocabulary_size_ger)


# In[75]:




class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        
    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        hidden = torch.zeros(self.n_layers, 1, self.hidden_size)
        if torch.cuda.is_available(): hidden = hidden.cuda()
        return hidden


# In[76]:




class MultiplicativeAttn(nn.Module):
    def __init__(self,  hidden_size, max_length=MAX_LENGTH):
        super(MultiplicativeAttn, self).__init__()
        
        
        self.hidden_size = hidden_size
        

        self.attn = nn.Linear(self.hidden_size, hidden_size)


    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = torch.zeros(seq_len) # B x 1 x S
        if torch.cuda.is_available(): attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        
      #print(hidden.size(),encoder_output.size())
      #energy = hidden.dot(encoder_output[0])
      #energy=torch.dot(hidden,encoder_output)
      energy=torch.mm(hidden,torch.t(encoder_output))
      return energy
        


# In[77]:


class ScaledDotProdAttn(nn.Module):
    def __init__(self,  hidden_size, max_length=MAX_LENGTH):
        super(ScaledDotProdAttn, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.attn = nn.Linear(self.hidden_size, hidden_size)


    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = torch.zeros(seq_len) # B x 1 x S
        if torch.cuda.is_available(): attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
      
      energy=torch.mm(hidden,torch.t(encoder_output))/math.sqrt(hidden_size)
      return energy


# In[78]:


class MultAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(MultAttnDecoderRNN, self).__init__()
        
        # Keep parameters for reference
        #self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)
        

        self.attn = MultiplicativeAttn(hidden_size)

    
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        
        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights


# In[79]:


class ScaledDotProdAttnDecoderRNN(nn.Module):
    def __init__(self,  hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(ScaledDotProdAttnDecoderRNN, self).__init__()
        
        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)
        

        self.attn = ScaledDotProdAttn(hidden_size)
    
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        
        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights


# In[80]:




class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1,max_length=MAX_LENGTH):
        super(BahdanauAttnDecoderRNN, self).__init__()
        
        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = BahadanauAttn(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, word_input, decoder_context,last_hidden, encoder_outputs):
        # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        word_embedded = self.dropout(word_embedded)
        
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        
        # Final output layer
        output = output.squeeze(0) # B x N
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output,decoder_context, hidden, attn_weights


# In[81]:


class BahadanauAttn(nn.Module):
    def __init__(self, hidden_size, max_length=MAX_LENGTH):
        super(BahadanauAttn, self).__init__()
        
        
        self.hidden_size = hidden_size
        

        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = torch.zeros(seq_len) # B x 1 x S
        if torch.cuda.is_available(): attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        
        energy = self.attn(torch.cat((hidden, encoder_output), 1))
        #energy = self.other.dot(energy)
        
        energy=torch.mm(self.other,torch.t(energy))
      
        return energy


# In[83]:


def indexesFromSentence(lang, sentence):
    l=list()
    if(lang=='eng'):
        
        for word in sentence.split(' '):
            if(word in vocabulary_eng):
                l.append(word2idx_eng[word])
            else:
                l.append(word2idx_eng["UNK"])
        
        #return [word2idx_eng[word] for word in sentence.split(' ')]
    else:
        for word in sentence.split(' '):
            if(word in vocabulary_ger):
                l.append(word2idx_ger[word])
            else:
                l.append(word2idx_ger["UNK"])
        #return [word2idx_eng[word] for word in sentence.split(' ')]
        
        
        #return [word2idx_ger[word] for word in sentence.split(' ')]
    return l


# In[84]:



def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1).to(device)


# In[85]:


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence("eng", pair[0]).to(device)
    target_tensor = tensorFromSentence("ger", pair[1]).to(device)
    return (input_tensor, target_tensor)


# In[86]:


teacher_forcing_ratio = 0.5
clip = 5.0

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word
    #print("hi1")
    # Get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Run words through encoder
    encoder_hidden = encoder.initHidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    # Prepare input and output variables
    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_context = torch.zeros(1, decoder.hidden_size)
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    if torch.cuda.is_available():
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        
        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di] # Next target is next input
            
       

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            
            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = torch.LongTensor([[ni]]) # Chosen word is next input
            if torch.cuda.is_available(): decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token: break
    #print(decoder_output.size(),decoder_context.size(), decoder_hidden.size(), decoder_attention.size())
    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    #return loss.data[0] / target_length
    return loss.item() / target_length


# In[87]:


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[ ]:





# In[88]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# In[89]:


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=40000, learning_rate=0.001):
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0].to(device)
        target_tensor = training_pair[1].to(device)

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    show_plot(plot_losses)


# In[90]:


def evaluate(encoder, decoder,sentence, max_length=MAX_LENGTH):
    input_variable = tensorFromSentence("eng", sentence)
    input_length = input_variable.size()[0]
    
    # Run through encoder
    encoder_hidden = encoder.initHidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([[SOS_token]]) # SOS
    decoder_context = torch.zeros(1, decoder.hidden_size)
    if torch.cuda.is_available():
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    decoder_hidden = encoder_hidden
    
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    
    # Run through decoder
    for di in range(max_length):
       # print(decoder_output.size(),decoder_context.size(), decoder_hidden.size(), decoder_attention.size())
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            #print(ni)
            decoded_words.append(idx2word_ger[ni.item()])
            
        # Next input is chosen word
        decoder_input = torch.LongTensor([[ni]])
        if torch.cuda.is_available(): decoder_input = decoder_input.cuda()
    
    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]


# In[91]:


print(vocabulary_size_eng)


# In[92]:


hidden_size = 256
encoder = EncoderRNN(vocabulary_size_eng, hidden_size)
encoder.to(device)
if(attn=="multiplicative"):
  decoder = MultAttnDecoderRNN(hidden_size, vocabulary_size_ger)
elif(attn=="scaledDotProd"):
  decoder = ScaledDotProdAttnDecoderRNN(hidden_size, vocabulary_size_ger)
elif(attn=="Bahadanau"):
  decoder = BahdanauAttnDecoderRNN(hidden_size, vocabulary_size_ger)
  
decoder.to(device)
trainIters(encoder, decoder, 120000, print_every=1000) #30 epochs (#training examples *30 =120000) 


# In[98]:


def evaluate_randomly(encoder,decoder):
    pair = random.choice(pairs)
    
    output_words, decoder_attn = evaluate(encoder,decoder,pair[0])
    output_sentence = ' '.join(output_words)
    
    print('>', pair[0])
    print('=', pair[1])
    print('<', output_sentence)
    print('')


# In[ ]:





# In[99]:


evaluate_randomly(encoder,decoder)


# In[100]:


##Testing
from nltk.translate.bleu_score import sentence_bleu

source_file = open("data/newstest2014-deen-ref.en.sgm", "r")
target_file = open("data/newstest2014-deen-ref.de.sgm", "r")
test_eng = source_file.readlines()
test_ger = target_file.readlines()


# In[101]:


test_pairs=list()
for i in range(len(test_eng)):
    test_pairs.append([normalizeString(test_eng[i]),normalizeString(test_ger[i])])


# In[103]:


score=0
for i in range(100):
    output_words, decoder_attn = evaluate(encoder,decoder,test_pairs[i][0])
    output_sentence = ' '.join(output_words)
    score += sentence_bleu(test_pairs[i][1], output_sentence)
score=score/len(test_eng)


# In[104]:


print(score)


# In[ ]:




