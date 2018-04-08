'''
Created on 2018. 4. 8.

@author: cdy-yoga
'''

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
from test.test_unicode_file_functions import filenames
from docutils.nodes import topic

all_letters = string.ascii_letters + ".,;'-"
n_letters = len(all_letters) + 1 # plus EOS marker

def findFiles(path): return glob.glob(path)

# Trun an unicode string to plaun ASCII, 

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
        )
    
# Read a file and split in to lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    
    return [unicodeToAscii(line) for line in lines]
    
# Build the category_Lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines
        
n_categories = len(all_categories)
    
print('# categories:', n_categories, all_categories)
print(unicodeToAscii("O'Néàl"))


'''
This network extends the last tutorial's RNN with an extra argument for the category tensor
which is concatenated along with the others. The category tensor is a one-hot veotr just like
the letter input.

We will interpret the output as the probability of the next letter. When sampling, the most
likely output letter is used as the next input letter.

i added a second linear layer o2o (after combining hidden and ouptut)to give it moore muscle
to work with. There's also a dropout layer, which randomly zeros parts of its input with a 
given probability (hear 0.1) abd is usually used to fuzz inputs to prevent overfitting. Here
we're using it towrads the end of the network to puerposly and some chaos and increase
sampling variety.

'''

""" Createing the network"""
 
import torch
import torch.nn as nn
from torch.autograd import Variable
 
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
         
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax()
        
    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)   # why combine
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
    
    
# Training

import random

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random live from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

'''
For each timestep (that is, for each letter in training word) the inputs of the network will be 
(category, current letter, hidden state) and the outpus will be
(next letter, next hidden state) So for each training set, we'll need the category, 
a set of input letters, and a set of output/target letters.

since we are predicting the next letter from the currnt letter for each timestep, the letter
pairs are groups of consecutive letter from the line -e.g for "ABCD<EOS>" we would create
("A", B") ("B","C"), ("C", "D"), ("D", "EOS").

the category tensor is a one-hot tensor of size <1 x n_categories>. When training we feed it 
to the network at every timestep - this is a designe choice, it could have been included as 
part of initial hidden state or some other strategy.
'''

# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of fist to last letters (not includeing EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    # data_lenth x 1 x data_dim
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

'''
For convenience during training we'll make a 'randomTrainingExample', function that fetches
a random(category, line) pair and turns them into the required(category, input, target)
tensor.
'''

# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = Variable(categoryTensor(category))
    input_line_tensor = Variable(inputTensor(line))
    target_line_tensor = Variable(targetTensor(line))
    return category_tensor, input_line_tensor, target_line_tensor


""" 
Training the Network

In contrast to classification, where only the last output is used, we ar making a prediction at
every step, so we are calculating loss at every step

the magic of autograd allows you to simply sum these losses at each step and call backward 
at the end.

"""

criterion = nn.NLLLoss()

learning_rate = 0.0005

def train(category_tensor, input_line_tensor, target_line_tensor):
    hidden = rnn.initHidden()
    
    rnn.zero_grad()
    
    loss = 0
    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i])
        
    loss.backward()
    
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)
        
    return output, loss.data[0] / input_line_tensor.size()[0]

'''
To keep track of how long training takes I am adding a 'timeSince(timestamp) function
which returns a human readable string:
'''


import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds' % (m, s)

'''
training is business as usual - call train a bunch of times and wait a few minutes, printing
the current time and loss every 'print_every' examples, and keeping stor of an average
loss per 'plot_every' examples n 'all_losses' for plotting later.
'''

rnn = RNN(n_letters, 128, n_letters)

n_iters = 1000
print_every = 50
plot_every = 5
all_losses = []
total_loss = 0 # reset every plot_every iters

start = time.time()

for iter in range(1, n_iters +1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss
    
    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter/n_iters*100, loss))
        
    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0
        
        

import matplotlib.pyolot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

"""
# sampling the network
to sample we give the network a letter and ask what the next one is, feed that in as the 
next letter, and repeat untile the EOS token

- create tensors for input categoru, starting letter, and empty hidden state
- create a string 'output_name' with the starting letter
- up to a maximum output lentgh,
  - feed the current letter to the network
  - get the next letter frm highest output, and next hidden state
  - if the letter is EOS, stop here
  - if a regular letter, add to 'output_name' and continue
- return the final name
"""

max_length = 20

# sample from a category and starting letter
def sample(category, start_letter  = 'A'):
    category_tensro = Variable(categoryTensor(category))
    input = Variable(inputTensor(start_letter))
    hidden = rnn.initHidden()
    
    output_name = start_letter
    
    for i in range(max_length):
        output, hidden = rnn(category_tensor, input[0], hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        if topi == n_letters - 1:
            break
        else:
            letter = all_letters[topi]
            output_name += letter
        input = Variable(inputTensor(letter))
        
    return output_name

# Get multiple sampels from one category and multiple starting letters
def samples(cataegory, start_letters = 'ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))
        
samples('Russian', 'RUS')   # maybe 'Namses//Russian' is correct if not error can be occur

samples('German', 'GER')

samples('Spanish', 'SPA')

Samples('Chinese', 'CHI')

        
        
    




    