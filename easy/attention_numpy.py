import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    print("e:", e_x.shape)
    print(np.sum(e_x, axis=-1, keepdims=True).shape)
    result = e_x / np.sum(e_x, axis=-1, keepdims=True)
    print("r:", result.shape)
    return result

def attention(x):
    n, d = x.shape
    Wq = np.random.rand(d, d)
    Wk = np.random.rand(d, d)
    Wv = np.random.rand(d, d)
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    A = q @ k.T
    A = A / np.sqrt(d)
    A_hat = softmax(A)
    output = A_hat @ v
    print(output.shape) # n, d


def multi_head_attention(x, head_n=16):
    n, d = x.shape
    assert d % head_n == 0
    Wq = np.random.rand(d, d)
    Wk = np.random.rand(d, d)
    Wv = np.random.rand(d, d)
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    q = np.reshape(q, (n, head_n, d // head_n))
    k = np.reshape(k, (n, head_n, d // head_n))
    v = np.reshape(v, (n, head_n, d // head_n))
    print("q.shape:", q.shape)
    q = np.transpose(q, (1, 0, 2))  # head_n, n, d // head_n
    k = np.transpose(k, (1, 0, 2))
    v = np.transpose(v, (1, 0, 2))
    print("q.shape:", q.shape, np.transpose(k, (0, 2, 1)).shape)
    A = q @ np.transpose(k, (0, 2, 1))
    print("A.shape:", A.shape)
    A = A / np.sqrt(d // head_n)
    print("A.shape:", A.shape)
    A_hat = softmax(A) # head_n, n, n
    print("A.shape:", A.shape)
    output = A_hat @ v # head_n, n, d // head_n
    output = np.transpose(output, (1, 0, 2))    # n, head_n, d // head_n
    output = np.reshape(output, (n, d)) 
    print(output.shape) # n, d
    

if __name__ == "__main__":
    attention(np.random.rand(512, 768))
    multi_head_attention(np.random.rand(512, 768))
