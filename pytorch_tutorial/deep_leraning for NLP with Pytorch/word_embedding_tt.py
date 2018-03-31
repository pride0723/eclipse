
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lib2to3.fixes.fix_input import context

torch.manual_seed(1)

word_to_ix = {"hello":0, "word": 1}
embeds = nn.Embedding(2,5)  # 2 words in vocab, 5 demensional embeddings
lookup_tensor = torch.LongTensor([word_to_ix["hello"]])  # learnable parameters?
hello_embed = embeds(autograd.Variable(lookup_tensor))
print(hello_embed)

#print(test_sentence)

# Recall that in aan n-gram language model, given a sequence of words W, 
# we want to compute
# P(w_i|w_(i-1), w_(i-2), ... w_(i-n+1))
# where w_i is the i th word of the sequence.

# in this exampel, we will compute the loss function on some training exampels
# and update the parameters with backpropagation


CONTEXT_SIZE = 2
EMBEDDINT_DIM = 10


test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# we should tokienize the input, but we will ignore that for now
# build a list of tuples. Each tuple is ([ word_i-2, word_i-1], target word)
trigrams = [([test_sentence[i], test_sentence[i+1]], test_sentence[i+2])
            for i in range(len(test_sentence) -2)]
# print the first 3 , just so you can see what they look like


#print(test_sentence.__sizeof__())
#print(trigrams.__sizeof__())
print(trigrams[:3])  # tuple structure is so complicated

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size*embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1,-1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDINT_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for each in range(10):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        
        # Step 1. prepare the inputs to be passed to the model(i.e, turn the words
        # inot integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        
        # Step 2. Recall that torch *accumulates* gradients. Before passing in a 
        # new instance, you need to zero out the gradients form the old
        # instance
        model.zero_grad()
        
        # step 3. run the forward pass, getting log probabilities over next word
        # words
        log_probs = model(context_var)
        
        # step 4. Compute your loss function. (Again, torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))
        
        # Step 5. Do the backward pass an update the gradient
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data
    losses.append(total_loss)
print(losses)


CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from 'raw_text', we deduplicate the arry
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word:i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) -2):
    context = [raw_text[i-2], raw_text[i-1],
               raw_text[i+1], raw_text[i+2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])

class CBOW(nn.Module):
    def __init__(self):
        pass
    def forward(self, inputs):
        pass

# create your model and train, here are some functions to help you make
# the data ready for use by your module

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

make_context_vector(data[0][0], word_to_ix)



