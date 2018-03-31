
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

# Create a torch.Tensor object with the given data. It is a 1D vector
V_data = [1., 2., 3.]
V = torch.Tensor(V_data)
print(V)

# Create a matrix
M_data = [[1., 2., 3.], [4., 5., 6.]]
M = torch.Tensor(M_data)
#print(M)

# Create a 3D tensor of size 2x2x2
T_data = [[[1., 2.], [3., 4.]],
          [[5., 6.], [7., 8.]]]
T = torch.Tensor(T_data)
#print(T)


print(V[0])
print(M[0])
print(T[0])


x = torch.randn((3, 4, 5))
print(x)

x = torch.Tensor([1., 2., 3.])
y = torch.Tensor([4., 5., 6.])
z = x+y
print(z)


# By defalut, it concatenates aling the first axis( concatenates rows)
x_1 = torch.randn(2,5)
y_1 = torch.randn(3,5)

print(x_1)
print(y_1)
z_1 = torch.cat([x_1, y_1])

print(z_1)
  

# Concatenate columns:
x_2 = torch.randn(2,3) 
y_2 = torch.randn(2, 5)
# second arg, specifies which axis to conca aling
z_2 = torch.cat([x_2, y_2],1)
print(z_2)

# if your tensor are not compatible, torch will complain. Un comment to see the
#torch.cat([x_1, x_2])

x = torch.randn(2, 3, 4)
print(x)

print(x.view(2, 12))
# same as aboce. if one of the dimensions is -1, its size can be inferred
print(x.view(2, -1))


# Variable wrap tensor objects
x = autograd.Variable(torch.Tensor([1., 2., 3]), requires_grad=True)
print(x.data)

# You can also do all the same opretaions you did with tensors with Variables.
y = autograd.Variable(torch.Tensor([4., 5., 6.]), requires_grad = True)
z = x+y
print(z.data)

# But z knows somthing extre

print(z.grad_fn)

s = z.sum()
print(s)
print(s.grad_fn)


s.backward()
print(x.grad)

x = torch.randn((2,2))
y = torch.randn((2,2))
z = x+y

var_x = autograd.Variable(x, requires_grad = True)
var_y =  autograd.Variable(y, requires_grad = True)

var_z = var_x + var_y
print(var_z.grad_fn)

var_z_data = var_z.data

new_var_z = autograd.Variable(var_z_data)

print(new_var_z.grad_fn)




