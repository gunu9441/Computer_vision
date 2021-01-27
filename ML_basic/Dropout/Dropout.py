import numpy as np

p = 0.5                   # probability of keeping a unit activate. if p is higher, less dropout will occur.
X1 = np.random.randn(3,3) # Input
W1 = np.random.randn(3,2) # Weight

print('X1 \n {}'.format(X1))
print('W1 \n {}'.format(W1))

H1 = np.maximum(0, np.dot(X1, W1)) # np.dot(X1, W1) 배열 모든 원소가 0과 비교(ReLU)
print('H1 \n {}'.format(H1))

print(H1.shape)
U1 = np.random.rand(*H1.shape)     # H1 크기의 0~1 정규분포 U1
print('U1 \n {}'.format(U1))

U1_check = U1 < p                  # U1가 p보다 작으면 True = 1, 아니면 False = 0
print('U1_check \n {}'.format(U1_check))

H1 *= U1_check                     # dropout

print('H1 \n {}'.format(H1))

