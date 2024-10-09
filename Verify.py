import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from main import x_test, y_test
# 加载模型
from main import LinearRegressionModel

model = LinearRegressionModel()
model.load_state_dict(torch.load('model.pth'))

# 将模型移动到GPU（如果有可用的GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 将测试数据移到GPU
x_test = x_test.to(device)
y_test = y_test.to(device)

# 模型测试阶段不需要梯度计算
model.eval()
with torch.no_grad():
    # 使用模型对测试集进行预测
    test_predictions = model(x_test)

# 将预测值和真实值转换为CPU上的numpy数组以进行误差计算
test_predictions = test_predictions.cpu().numpy()
y_test = y_test.cpu().numpy()

# 计算MAE, MSE 和 RMSE
mae = mean_absolute_error(y_test, test_predictions)
mse = mean_squared_error(y_test, test_predictions)
rmse = np.sqrt(mse)

# 打印评估指标
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# 可视化真实值和预测值对比
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(y_test, label='True Values')
plt.plot(test_predictions, label='Predictions')
plt.title('True Values vs Predictions')
plt.xlabel('Sample index')
plt.ylabel('Median House Value')
plt.legend()
plt.grid(True)
plt.show()
