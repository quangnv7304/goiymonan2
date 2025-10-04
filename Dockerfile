# ---- Giai đoạn 1: Xây dựng môi trường ----
# Bắt đầu từ một image Python 3.9 gọn nhẹ, chính thức
FROM python:3.9-slim

# Đặt thư mục làm việc mặc định bên trong container
WORKDIR /app

# Sao chép file requirements.txt vào trước để tận dụng cache của Docker
# Nếu file này không đổi, Docker sẽ không cần cài lại thư viện ở các lần build sau
COPY requirements.txt .

# Chạy lệnh pip để cài đặt tất cả các thư viện đã được định nghĩa
# --no-cache-dir giúp giảm kích thước cuối cùng của image
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ các file code còn lại (server.py, agent.py, q_table...) vào container
COPY . .

# Cho Docker biết rằng ứng dụng bên trong container sẽ lắng nghe trên cổng 8000
EXPOSE 8000

# Lệnh mặc định sẽ được thực thi khi container khởi động
# Chạy server FastAPI bằng uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]