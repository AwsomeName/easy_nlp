import numpy as np

# 生成一些随机数据
np.random.seed(42)
X = np.random.rand(100, 2)  # 100个样本，2个特征
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # 二分类任务

# 定义一个简单的MLP，包含一个隐藏层
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        self.hidden = np.maximum(0, np.dot(X, self.W1) + self.b1)  # ReLU激活函数
        self.output = np.dot(self.hidden, self.W2) + self.b2
        return self.output

    def backward(self, X, y, learning_rate=0.01):
        # 计算梯度
        d_output = self.output - y
        d_W2 = np.dot(self.hidden.T, d_output)
        d_b2 = np.sum(d_output, axis=0)
        d_hidden = np.dot(d_output, self.W2.T) * (self.hidden > 0)
        d_W1 = np.dot(X.T, d_hidden)
        d_b1 = np.sum(d_hidden, axis=0)

        # 更新权重
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2

    def train(self, X, y, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

# 初始化MLP
mlp = MLP(input_size=2, hidden_size=10, output_size=1)

# 训练模型
mlp.train(X, y, epochs=1000, learning_rate=0.01)

# 预测
predictions = (mlp.forward(X) > 0.5).astype(int)

# 打印准确率
accuracy = np.mean(predictions == y)
print(f"准确率：{accuracy:.2f}")
