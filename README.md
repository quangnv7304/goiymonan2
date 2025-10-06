# SARSA training helper

Hướng dẫn nhanh để huấn luyện SARSA agent trong thư mục `q_learning`.

Yêu cầu:
- Python
- Virtualenv (có sẵn `venv` trong repo) hoặc environment của bạn
- Các package trong `requirements.txt` (gym, numpy, ...)

Chạy training dài (ví dụ 10k episodes):

```powershell
Set-Location -Path 'E:\goiymonan1\q_learning'
E:\goiymonan1\q_learning\venv\Scripts\python.exe E:\goiymonan1\q_learning\train.py --episodes 10000 --save-every 500
```

Ghi chú:
- Nếu `Activate.ps1` báo lỗi do `pyvenv.cfg` trỏ tới Python cũ, chạy python trực tiếp như trên.
- Nếu bạn dùng Gymnasium (thay vì Gym), hãy thông báo để tôi cập nhật code tương thích hoàn toàn.
