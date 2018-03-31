


import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100,10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

## nn.Sequential is module which contains otehr modulse, 

model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out),
    )
    
print(model)

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4

## using optim
optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate)


for t in range(500):
    y_pred = model(x)
    
    loss = loss_fn(y_pred,y)
    print(t, loss.data[0])
    
    ## not using optim
    # model.zero_grad()
    # loss.backward()
    # for param in model.parameters():
    #    param.data -= learning_rate* param.grad.data
    
    # using optim
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    




print(model.parameters())  # how to extract parameters from here?