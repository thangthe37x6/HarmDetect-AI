# harmscan ai

hướng dẫn cài đặt và chạy dự án harmscan ai

## yêu cầu hệ thống

- python 3.10 hoặc cao hơn
- node.js và npm
- postgresql

## hướng dẫn cài đặt

### 1. tạo thư mục dự án

```bash
mkdir Harmscan_AI
cd Harmscan_AI
```

### 2. thiết lập môi trường python

tạo môi trường ảo với tên `process_video`:

```bash
python -m venv process_video
```

### 3. kích hoạt môi trường ảo

**windows:**
```bash
process_video\Scripts\activate
```

**macos/linux:**
```bash
source process_video/bin/activate
```

### 4. cài đặt thư viện python

```bash
pip install -r requirements.txt
```

### 5. cài đặt dependencies cho server

chuyển đến thư mục server và cài đặt các package npm:

```bash
cd server
npm install
```

### 6. tải model

do file model `best_violence_model.pt` có dung lượng lớn (79.20 MB), bạn cần tải về từ google drive:

**[📥 tải model tại đây](https://drive.google.com/drive/folders/1fq2CfY75H4PTY2ZcbCwTxbX1m9cFl8_h?usp=sharing)**

sau khi tải về, đặt file `best_violence_model.pt` vào thư mục `server/`:

```
server/
  ├── best_violence_model.pt  ← đặt file model vào đây
  ├── server.js
  └── ...
```

### 7. cấu hình môi trường

tạo file `.env` trong thư mục `server` và thêm các thông tin sau:

```env
OPENAI_API_KEY=your_openai_api_key_here
password_portSQL=your_postgres_password_here
```

**lưu ý:** thay thế `your_openai_api_key_here` và `your_postgres_password_here` bằng thông tin thực tế của bạn.

### 8. chạy server

```bash
node server.js
```

## hoàn tất

server đã sẵn sàng hoạt động! 🚀

## liên hệ

nếu có bất kỳ vấn đề gì, vui lòng tạo issue trong repository này.

