[🇬🇧 English version](README.md)

# MoodTune

Gợi ý nhạc theo cảm xúc: người dùng nhập một đoạn văn bản, backend dùng một AI engine (rule-based lexicon + MLP có self-attention, học online từ feedback) để đoán cảm xúc, rồi tìm nhạc phù hợp qua Jamendo API (free, không cần đăng nhập).

## Cấu trúc

- `backend/` — Flask API (Python). Engine cảm xúc (`emotion_mlp.py`, `lexicon.py`), Thompson Sampling bandit để học gu nhạc (`bandit.py`), tìm nhạc Jamendo + audio feature analysis (`audio_features.py`), entrypoint `app.py`.
- `frontend/` — single-page app (HTML/CSS/JS thuần, không build step) trong `index.html`.

## Chạy backend

Yêu cầu Python 3.x.

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Backend chạy ở `http://localhost:5005`, API base path là `/api` (xem `/api/health` để kiểm tra).

Biến môi trường (đều có giá trị mặc định, không bắt buộc set khi chạy local):

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `MOODTUNE_SECRET_KEY` | `moodtune_secret_2024` | Flask `secret_key`. Nên set giá trị riêng khi deploy thật. |
| `MOODTUNE_FRONTEND` | `https://anhtaictv.me` | Origin của frontend, dùng để cấu hình CORS. |
| `JAMENDO_CLIENT_ID` | `cf31dbfd` | Client ID gọi Jamendo API. |

Để chạy bằng pm2 (production), dùng config có sẵn:

```bash
pm2 start backend/ecosystem.config.js
```

## Chạy frontend

Mở trực tiếp `frontend/index.html` bằng trình duyệt (hoặc qua `http://localhost:5500` bằng static server tuỳ ý). Khi chạy ở `localhost`/`file://`, frontend tự gọi thẳng backend tại `http://localhost:5005/api` — không cần cấu hình thêm.

Khi deploy production, frontend gọi `/api` (relative path), nên cần một reverse proxy (IIS/Nginx) trỏ `/api` sang backend Flask. Cấu hình IIS mẫu xem `frontend/web.config`.

## Dữ liệu model

`weights.npz`, `weights_meta.json`, `weights_replay.json`, `dynamic_vocab.json` trong `backend/` là trạng thái đã học của model (vocab, trọng số MLP, replay buffer cho online learning). `feedback_log.jsonl` lưu lịch sử predict/feedback/hành vi nghe nhạc, dùng để bandit học gu nhạc và thống kê `/api/stats`.
