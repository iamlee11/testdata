import pandas as pd
import os

# Đường dẫn đến file dữ liệu (cập nhật đúng đường dẫn)
file_path = r"d:\OneDrive - vnu.edu.vn\New folder\household_power_consumption.txt"

# Kiểm tra file tồn tại
if os.path.exists(file_path):
    print("File exists! Đang đọc dữ liệu...")
else:
    raise FileNotFoundError("File not found! Kiểm tra lại đường dẫn.")

# Đọc file với pandas
data = pd.read_csv(file_path, sep=';', na_values=['?'], low_memory=False)

# Kiểm tra thông tin cơ bản của dữ liệu
print("Thông tin dữ liệu:")
print(data.info())
print(data.head())
# 1. Kết hợp Date và Time thành Datetime
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')

# 2. Chọn cột quan trọng và làm sạch dữ liệu
data = data[['Datetime', 'Global_active_power']].dropna()

# 3. Chuyển Global_active_power sang kiểu số thực
data['Global_active_power'] = data['Global_active_power'].astype(float)

# 4. Tổng hợp dữ liệu theo ngày
data_daily = data.groupby(data['Datetime'].dt.date)['Global_active_power'].mean()
data_daily.index = pd.to_datetime(data_daily.index)

# Hiển thị thông tin sau khi xử lý
print(data_daily.head())
import matplotlib.pyplot as plt

# Vẽ biểu đồ xu hướng
plt.figure(figsize=(12, 6))
plt.plot(data_daily.index, data_daily.values, label='Daily Average Power', color='blue')
plt.title("Daily Average Global Active Power")
plt.xlabel("Date")
plt.ylabel("Power (kW)")
plt.legend()
plt.grid()
plt.show()
# Mô tả thống kê cơ bản
print("Thống kê cơ bản:")
print(data_daily.describe())
from sklearn.preprocessing import MinMaxScaler

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_daily.values.reshape(-1, 1))
import numpy as np

# Tạo chuỗi thời gian
def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

# Sử dụng 30 ngày để dự đoán ngày tiếp theo
time_steps = 30
X, y = create_sequences(data_scaled, time_steps)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
import torch
import torch.nn as nn
import torch.optim as optim

# Định nghĩa mô hình LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM trả về (output, (hidden, cell))
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Lấy output cuối cùng
        return out

# Khởi tạo mô hình
input_size = 1
hidden_size = 50
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size)

# Định nghĩa hàm mất mát và bộ tối ưu hóa
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Chuyển dữ liệu sang PyTorch tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Huấn luyện mô hình
epochs = 20
batch_size = 32
for epoch in range(epochs):
    model.train()  # Đặt mô hình ở chế độ huấn luyện
    for i in range(0, len(X_train_tensor), batch_size):
        # Lấy batch
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]

        # Dự đoán
        outputs = model(X_batch)

        # Tính toán mất mát
        loss = criterion(outputs, y_batch)

        # Cập nhật trọng số
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Kiểm tra trên tập validation
    model.eval()  # Đặt mô hình ở chế độ đánh giá
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")
import matplotlib.pyplot as plt

epochs = range(1, 21)  # Từ 1 đến 20
loss = [0.003783572232350707, 0.0039148712530732155, 0.004058351274579763, 0.004201008006930351, 0.00459169689565897,
        0.005865336861461401, 0.010582813993096352, 0.022637832909822464, 0.0116026746109128, 0.0183627400547266,
        0.01483486220240593, 0.016828933730721474, 0.015551619231700897, 0.016095926985144615, 0.01564435474574566,
        0.0157187357544899, 0.015531138516962528, 0.015476159751415253, 0.015362611040472984, 0.015279262326657772]
val_loss = [0.009645894169807434, 0.008984687738120556, 0.008424846455454826, 0.008018909022212029, 0.007599642965942621,
            0.00711705069988966, 0.006456232629716396, 0.006110014859586954, 0.006161208264529705, 0.006042888853698969,
            0.006038615480065346, 0.005998432636260986, 0.005983111914247274, 0.005961045157164335, 0.0059458231553435326,
            0.005930264014750719, 0.0059171319007873535, 0.005904827266931534, 0.005893772002309561, 0.005883586592972279]

plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Validation Loss')
plt.show()
# X_test_tensor: Dữ liệu đầu vào để kiểm tra
model.eval()  # Chuyển sang chế độ dự đoán
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

# Hiển thị dự đoán
print(predictions)
import matplotlib.pyplot as plt

# Giả sử y_test là giá trị thực tế và predictions là giá trị dự đoán
plt.figure(figsize=(14, 7))
plt.plot(y_test, label="Thực tế", color='blue')
plt.plot(predictions, label="Dự đoán", color='orange')
plt.title("So sánh giá trị thực tế và dự đoán")
plt.xlabel("Thời gian")
plt.ylabel("Công suất tiêu thụ (kW)")
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Tính MSE và RMSE
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
import matplotlib.pyplot as plt

# Vẽ biểu đồ so sánh
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Thực tế', color='blue', linewidth=2)
plt.plot(predictions, label='Dự đoán', color='orange', linewidth=2)
plt.title("So sánh giá trị thực tế và dự đoán")
plt.xlabel("Thời gian")
plt.ylabel("Công suất tiêu thụ (kW)")
plt.legend()
plt.grid(True)
plt.show()
import numpy as np

# Tìm chỉ số của các điểm sai số lớn nhất
errors = np.abs(y_test - predictions)
max_error_indices = np.argsort(errors)[-5:]  # 5 điểm có sai số lớn nhất

# In các điểm sai số lớn nhất
for idx in max_error_indices:
    print(f"Thời điểm: {idx}, Thực tế: {y_test[idx]}, Dự đoán: {predictions[idx]}, Sai số: {errors[idx]}")
import numpy as np

# Chuyển y_test và predictions sang mảng numpy để xử lý
y_test = np.array(y_test).reshape(-1)  # Chuyển y_test thành vector 1 chiều
predictions = np.array(predictions).reshape(-1)  # Tương tự cho predictions

# Tính sai số
errors = np.abs(y_test - predictions)

# Tìm 5 điểm có sai số lớn nhất
max_error_indices = np.argsort(errors)[-5:]  # Lấy chỉ số 5 điểm có sai số lớn nhất

# Hiển thị các điểm sai số lớn nhất
for idx in max_error_indices:
    print(f"Thời điểm: {idx}, Thực tế: {y_test[idx]}, Dự đoán: {predictions[idx]}, Sai số: {errors[idx]}")
print("Kích thước y_test:", y_test.shape)
print("Kích thước predictions:", predictions.shape)
print(f"Sai số nhỏ nhất: {np.min(errors)}")
print(f"Sai số lớn nhất: {np.max(errors)}")
# Giả sử `original_data` là tập dữ liệu gốc
for idx in [23, 241, 258, 15, 221]:
    print(f"Thời điểm {idx}, Dữ liệu gốc: {original_data[idx]}")
import pandas as pd

# Đọc dữ liệu từ tệp CSV (giả sử tên tệp là "data.csv")
original_data = pd.read_csv("data.csv")

# Kiểm tra dữ liệu
print(original_data.head())
# Giả sử bạn đã chia dữ liệu thành train/test
# Lấy dữ liệu gốc tương ứng với y_test
original_data_test = original_data.iloc[test_indices]  # test_indices là chỉ số của y_test trong dữ liệu gốc

# Lấy thông tin tại các thời điểm sai số lớn
for idx in [23, 241, 258, 15, 221]:
    print(f"Thời điểm {idx}, Dữ liệu gốc: {original_data_test.iloc[idx]}")
