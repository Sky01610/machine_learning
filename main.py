import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

# 加载加利福尼亚房价数据集
housing = fetch_california_housing()
data, target = housing.data, housing.target

# 数据标准化处理
scaler = StandardScaler()
data = scaler.fit_transform(data)

# 将数据转换为 PyTorch 张量
data = torch.tensor(data, dtype=torch.float32)
target = torch.tensor(target, dtype=torch.float32).view(-1, 1)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)


# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(data.shape[1], 1)  # 输入特征数为数据集的特征数

    def forward(self, x):
        return self.linear(x)


def train_model(device, x_train, y_train, x_test, y_test, batch_size, learning_rate):
    #检查是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 将数据移动到指定设备
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    model = LinearRegressionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    num_epochs = 100
    train_losses = []
    test_losses = []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()

        # 迭代所有数据集的批次
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        # 测试集上的损失
        model.eval()
        with torch.no_grad():
            test_predictions = model(x_test)
            test_loss = criterion(test_predictions, y_test)
            test_losses.append(test_loss.item())

        if (epoch + 1) % 10 == 0:
            print(
                f'Device: {device}, Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    end_time = time.time()
    training_time = end_time - start_time

    return model, train_losses, test_losses, training_time


# 设置批处理大小和学习率
batch_size = 32
learning_rate = 0.001  # 降低学习率

# 在GPU上训练
gpu_device = torch.device("cuda")
gpu_model, gpu_train_losses, gpu_test_losses, gpu_time = train_model(gpu_device, x_train, y_train, x_test, y_test,
                                                                     batch_size, learning_rate)
# Save the model's state_dict
torch.save(gpu_model.state_dict(), 'model.pth')

# 绘制训练损失随训练周期的变化图
plt.figure(figsize=(10, 5))
plt.plot(range(len(gpu_train_losses)), gpu_train_losses, label='GPU Training Loss')
plt.title('Training Loss over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 绘制测试损失随训练周期的变化图
plt.figure(figsize=(10, 5))
plt.plot(range(len(gpu_test_losses)), gpu_test_losses, label='GPU Test Loss')
plt.title('Test Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 打印训练时间
print(f"GPU training time: {gpu_time:.2f} seconds")
