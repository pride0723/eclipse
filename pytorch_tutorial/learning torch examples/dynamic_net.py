# -*- coding: utf-8 -*-


import torch
import random
from torch.autograd import Variable


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)  # 여기서 weight를 선언하는데 middle layer해당하는 것 하나만 선언?
        self.output_linear = torch.nn.Linear(H, D_out)   
        
    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0,3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)  # middle_layer의 경우 weight를 하나만 계속해서 사용 
        y_pred = self.output_linear(h_relu)            # __init__에서 선언 한것만 weight를 사용 하는 것이고 여기서 여러번 사용하더라도 위에서 선언한 하나의 weight 만 사용?
        return y_pred

N, D_in, H, D_out = 64, 1000, 100,10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

## nn.Sequential is module which contains otehr modulse, 

model = DynamicNet(D_in, H, D_out)
    



#loss_fn = torch.nn.MSELoss(size_average=False)
criterion = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4

## using optim
optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate)


for t in range(500):
    y_pred = model(x)
    
    # loss = loss_fn(y_pred,y)
    loss = criterion(y_pred,y)
    print(t, loss.data[0])

    # using optim
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    




print(model.parameters())  # how to extract parameters from here?