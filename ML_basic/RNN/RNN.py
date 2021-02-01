import numpy as np

#data input/output

data = open('input.txt', 'r').read()
chars = list(set(data))

# print(data)
# print(chars)

data_size, vocab_size = len(data), len(chars) #data_size => 문서 전체 길이  vocab vocab_size => 알파벳의 갯수이며 one-hot encoding에 사용

# print(data_size, vocab_size)

print(f'data size: {data_size} \nunique: {vocab_size}')

char_to_ix = {ch:i for i,ch in enumerate(chars)} #dict
# print(type(char_to_ix))
# print(char_to_ix)

ix_to_char = {i:ch for i,ch in enumerate(chars)}
# print(type(ix_to_char))
# print(ix_to_char)

#hyperparameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

#model 
Wxh = np.random.randn(hidden_size, vocab_size)  * 0.01 
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size)  * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))

def sample(hprev ,seed_ix, n):         
    x = np.zeros((vocab_size,1))   #(vocab_size, 1) 크기의 input x를 전부 0으로 생성
    x[seed_ix] = 1                 # x를 one hot encoding으로 만들어주기 위해 알맞은 곳에 1 넣기
    ixes = []

    for t in range(n):
        h = np.tanh(np.dot(Wxh,x) + np.dot(Whh,hprev) + bh)
        y = np.dot(Why,h) + by
        p = np.exp(y) / np.exp(y).sum()
        # print(p.shape)
        # print(p)
        # print(range(vocab_size))
        # print(p.ravel())
        # ix = np.random.choice()
        sampled_x = np.random.choice(range(vocab_size), p = p.ravel())
        print(range(vocab_size))
        print(sampled_x)
        x = np.zeros((vocab_size, 1))
        x[sampled_x] = 1
        print(x)
        ixes.append(x)
        print(ixes)
    return ixes
        
        

sample(np.random.randn(hidden_size,1), 0, 1)

    
