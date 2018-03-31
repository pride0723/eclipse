
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from _ast import Num


torch.manual_seed(1)

lin = nn.Linear(5,3) # nn.Linear(5,3) # maps from R^5 to R^3, paramters A, b
# data is 2x5. A maps from 5 to 3... can we map "data" under A?


print(lin)
data = autograd.Variable(torch.randn(2,5))
print(lin(data))



data = autograd.Variable(torch.randn(2,2))
print(data)
print(F.relu(data))


      
data = autograd.Variable(torch.randn(5))
print(data)
print(F.softmax(data))      
print(F.softmax(data).sum())
print(F.log_softmax(data)) # log softmax -> i don't now what it is




print('Example: Logitstic Regression Bag-of-words classifier')

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]


 
test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

# word_to_ix maps each woard in the vocab to a uniqu integer, which will be its
# index into the bag of words vector
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
            
print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2

class BoWClassifier(nn.Module): #inheriting from nn.Module!
        
    def __init__ (self, num_labels, vacab_size):
        # calls the init funtion of nn.Module. Dont get confused by syntax,
        # just alwars do it in an nn.Moudel
        super(BoWClassifier, self).__init__()
        
        # Define the parameters that you will need. In this case, we need A and B,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provieds the affine map.
        # Make sure you understand why the input dimension is VOCAB_size
        # and the output is num_lables!
        self.linear = nn.Linear(vacab_size, num_labels)
        
        # NOTE! The non-Linearity Long softmax does not have paramters! So we don't need
        # to worry about that here
        
    def forward(self, bow_vec):
        # pass the input through the linear layer,
        # then pass that through log_softmax.
        # manu non-linearities and other functions ar in torch.nn.functional'
        return F.log_softmax(self.linear(bow_vec))
    
def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for ward in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1,-1)

def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])

model=BoWClassifier(NUM_LABELS, VOCAB_SIZE)


# the model knows its paramters. the first output below is a, the second is b.
# Whenver you assign a component to a class Variable in the __init__ function
# of a module, which wad done with the line
# self.linear = nn.Linear(...)
# then through some python maginc from the pytorch devs, your module
# (in this case, BoWClassfier) will stor knowleadge of the nn.Linear's paramters


for param in model.parameters():
    print(param)


# To run the model, pass in a BoW vector, but wrapped in an autograd.Variable
sample = data[0]
bow_vector = make_bow_vector(sample[0], word_to_ix)
log_probs = model(autograd.Variable(bow_vector))
print(log_probs)
                             
label_to_ix = {"SPANISH":0, "ENGLISH":1}


"""
So lets train! To do this, we pass instance through to get log probabilities, compute a 
loss function, compute the gradient of the loss function, and then update the parameters
with a gradients step. loss functions are provided by torch in the nn package.nn.NLLLoss()
is the negative log likelihood loss we wnat. it also defines optimization functions in 
torch.optim. here we will just use SGD.

Note that the input to NLLLoss is a vector of log probabilities, and a target label.
is doesn;t compute the log probabilites for u. This is why the last layer of our network is log soeftmax.
The loss function nn.CrossEntropyLoss() it the smae as NLLLoss().
except it does the log softmax for you
"""


for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print(log_probs)

# print the matrix column correspoding to "creo"
print(next(model.parameters())[:, word_to_ix["creo"]])           

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Usally you wanr to pass over the training data several times.
# 100 is much bigger than on a real data set, but real dataset have more than
# two instance. Usally, somewhare between 5 and 30 epochs is reasonable.

for epoch in range(100):
    for instance, label in data:
        
        model.zero_grad()
        
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        target = autograd.Variable(make_target(label,label_to_ix))
        
        log_probs = model(bow_vec)
        
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print(log_probs)
    

print(next(model.parameters())[:, word_to_ix["lost"]])
            
