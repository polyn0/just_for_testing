# dictionary - dict()

dic = dict()
dic["name"] = "Heekyung"
dic["town"] = "Goyang city"
dic["job"] = "Assistant professor"
print dic

# class

class Student:
    def __init__(self, name):
        self.name = name
    def study(self, hard = False):
        if hard:
            print("%s 학생은 열심히 공부합니다."%self.name)
        else:
            print("%s 학생은 공부합니다."%self.name)

s = Student('Heekyung')
s.study()
s.study(hard = True)



## vector, matrices

#library load
import numpy as np

def print_val(x):
    print "Type:", type(x)
    print "Shape:", x.shape
    print "값:\n", x
    print " "
    
# rank 1 np array
x = np.array([1, 2, 3])
print_val(x)

x[0] = 5
print_val(x)
    
    
# rank 2 np array
y = np.array([[1,2,3], [4,5,6]])
print_val(y)

# rank 2 zeros
a = np.zeros((2,2))
print_val(a)

# rank 2 ones
b = np.ones((3,2))
print_val(b)

# rank 2 identity matrix
c = np.eye(3,3)
print_val(c)


# random matrix(uniform)
d = np.random.random((4,4))
print_val(d)

# random matrix(Gaussian)
e = np.random.randn(4,4)
print_val(e)



# np array indexing
f = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print_val(f)

g = f[:2, 1:3]
print_val(g)

# nth row 
row1 = f[1, :] # row 1
print_val(row1) 

## 벡터, 행렬 연산(1) - elementwise

m1 = np.array([[1,2], [3,4]], dtype = np.float64) # float
m2 = np.array([[5,6], [7,8]], dtype = np.float64)

# elementwise sum
print_val(m1 + m2)
print_val(np.add(m1, m2))

# element difference
print_val(m1 - m2)
print_val(np.subtract(m1, m2))

# elementwise product
print_val(m1 * m2)
print_val(np.multiply(m1, m2))

# elementwise division
print_val(m1 / m2) # float
print_val(np.divide(m1, m2))

# elementwise square root
print_val(np.sqrt(m1))


## 벡터, 행렬 연산(2) - inner product, transpose

m1 = np.array([[1,2], [3,4]])
m2 = np.array([[5,6], [7,8]])
v1 = np.array([9,10])
v2 = np.array([11,12])

print_val(m1)
print_val(m2)
print_val(v1)
print_val(v2)
print " "


# vector-vctor
print_val(v1.dot(v2))
print_val(np.dot(v1, v2))

# vector-matrix
print_val(m1.dot(v1))
print_val(np.dot(m1, v1))

# matrix-matrix
print_val(m1.dot(m2))
print_val(np.dot(m1, m2))
print " "


# transpose
print_val(m1)
print_val(m1.T)


## 벡터, 행렬 연산(3) - axis, zeros-like

# sum
m1 = np.array([[1,2], [3,4]])

print_val(np.sum(m1))
print_val(np.sum(m1, axis = 0));
print_val(np.sum(m1, axis = 1));


m2 = np.array([[1,2,3], [4,5,6]])

print m2
print m2.shape
print " "

print np.sum(m2)
print np.sum(m2, axis = 0)
print np.sum(m2, axis = 1)


#zeros-like
m3 = np.array([[1,2,3],
             [4,5,6],
             [7,8,9], 
             [10,11,12]])
m4 = np.zeros_like(m3)
print_val(m3)
print_val(m4)

## 그래프 그리기

import matplotlib.pyplot as plt
%matplotlib inline

# sin curve
x = np.arange(0,10,0.1)
y = np.sin(x)

plt.plot(x,y)
plt.show()

# 2 graphes
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)

plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('sin and cos')
plt.legend(['sin','cos'])
plt.show()


#subplot
plt.subplot(2,1,1)
plt.plot(x, y_sin)
plt.title('sin')

plt.subplot(2,1,2)
plt.plot(x,y_cos)
plt.title('cos')

plt.show()