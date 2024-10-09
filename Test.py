import torch
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn as nn
# 加载加利福尼亚房价数据集

housing = fetch_california_housing()
# 定义线性回归模型结构（与训练时一致）

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(8, 1)  # 8个特征（California Housing有8个特征）

    def forward(self, x):
        return self.linear(x)

# 加载模型
model = LinearRegressionModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # 设置为评估模式，禁用dropout等层

# 假设你有新的输入数据，格式如下（8个特征的一个样本或多个样本）
# 这是一个示例新数据
new_data = np.array([[8.3252, 41.0, 6.984127, 1.023809, 322.0, 2.555556, 37.88, -122.23]])

# 对新输入的数据进行标准化（与训练时一致的标准化器）
scaler = StandardScaler()
# 假设 `scaler.mean_` 和 `scaler.scale_` 是在训练期间保存的均值和缩放值
# 你可以在训练完成后保存标准化器，或者在加载数据时重新标准化整个数据集
scaler.fit(housing.data)  # 重新加载原始数据进行fit

# 标准化新数据
new_data_scaled = scaler.transform(new_data)

# 将新数据转换为 PyTorch 张量
new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

# 将模型和数据移动到 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
new_data_tensor = new_data_tensor.to(device)

# 使用模型进行预测
with torch.no_grad():  # 禁用梯度计算
    predictions = model(new_data_tensor)

# 将预测值转换为 numpy 数组，并打印
predictions = predictions.cpu().numpy()
print(f"Predicted house value: {predictions[0][0]:.4f}")

