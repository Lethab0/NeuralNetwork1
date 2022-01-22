import numpy 
import random
import math

def NN(m1 ,m2 , w1 ,w2, b):
    z = m1*w1 + m2*w2 + b
    return sigmoid(z)
    

def sigmoid(x):
    """The sigmoid function."""
    return 1.0/(1.0+numpy.exp(-x))


w1 = numpy.random.randn()
w2 = numpy.random.randn()
b = numpy.random.randn()



# derivative of Cost  function (prediction - target)

def cost(y):
    return (y -1)**2

num = NN(20 ,5 ,w1,w2,b)

print(num)
print(cost(num))

def num_slope(y):
    h = 0.00001
    return (cost(y+h) - cost(y))/h

print(num_slope(num))


def slope(b):
    return 2 *(b-4)


# we use a fraction of the slope else we overshoot
print(slope(4))

b = 0
for i in range(500000):
    b  =  b- .1*slope(b)
    print(b) 

#Cost function for linear regression
# The total cost funtion 
# Best to divide the cost by the number of datapoints to have a smaller number to work with





# You firtly look for the total cost(w,b) then look for the partial derivitives of  cost of w and b
# or just use a library for this
# dcdw(w,b) = ....
# dcdb(w,b)=...
# multipy by a a fraction negative number  so you dont overshoot and to move toward target instead of awway
#alpha = -.001
# 
# w= alpha*dcdw(w,b)
# b= alpha*dcdb(w,b)   
#








 
