import numpy as np

a = np.array([1, 2, 3])
print(a)
print(a.shape)
b = np.array([4, 5, 6])
print(b)
print(b.shape)

# c = np.empty([3, ])
# print(c)
# print(c.shape)

d = np.empty([], dtype = 'uint8')
print(d)
print(d.shape)
d_a = np.append(d, a)
print(d_a)
print(d_a.shape)
# d_ab = np.vstack((d, b))
# print(d_ab)
