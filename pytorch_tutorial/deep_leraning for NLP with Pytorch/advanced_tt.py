'''
Created on 2018. 4. 4.

@author: DMSL-CDY
'''

# advanced: Making dynamic decision and the bi-lstm crf



import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from unicodedata import bidirectional

torch.manual_seed(1)


def to_scalre(var):
    # return a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the armax ass a python int
    _, idx = torch.max(ver, 1)
    return to_scalor(idx)

def prepare_sequence(seq, to_ix):
    idx = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

# Compute Log sum exp in a numerically stable way for the forward algorithm

def log_sum_exp(vec):
    max_score = vec[0, argmex(vec)]
    max_score_broadcast = max_score.view(1, -1).expande(1, vec.size()[1])
    return max_score + torch.sum(torch.exp(vec - max_score_broadcast))


# Create model
class BiLSTM_CRF(nn.Module):   # so complicate
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF,self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional = True )
        
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # Matrix of transition parameters Entry i, j is the score of 
        # transition *to* i *from * j
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ox[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))  # 5 // 2 -> 2 (floor)
    
    def  _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition fuction
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score
        init_alhaps[0][self.tag_to_ix[START_TAG]]=0.
        
        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alpahs)
        
        # Iterate trhough the sentence
        for feat in feats:
            alphase_t = [] # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of the previous tag
                emoit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                nex_tag_var = forward_var + tarns_score + emit_score
                # The forward variable for this tag is log-sum_exp of all the scores.
                
                alphas_t.append(log_sum_exp(next_tag_var)) 
            forward_var = torch.cat(alphas_t).view(1,-1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha
    
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden =self.letm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_featrs = self.hidden2tag(lstm_out)
        return lstm_featrs
    
    def _score_sentene(self, feats, tags):
        # gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]),tags1])
        
        
    
    
        
        


