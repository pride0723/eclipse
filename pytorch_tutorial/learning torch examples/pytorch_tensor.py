
import torch
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is iput dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data

x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)

w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

learning_rate = 1e-6
for t in range(500):
    # forward pass: compute predicted y
    h = x.mm(w1)   # mm -> matrix multiplictaion = .dot() 
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    
    # Compute and print loss
    loss = (y_pred-y).pow(2).sum()
    print(t, loss)
    
    # Backward to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred -  y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h<0] = 0
    grad_w1 = x.t().mm(grad_h)
    
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2

