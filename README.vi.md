[🇬🇧 English version](README.md)

# MoodTune

**Phiên bản hiện tại: `v3.8`** — xem mục [Lịch sử phiên bản](#lịch-sử-phiên-bản) ở cuối trang để biết chi tiết từng bản.

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

## Lịch sử phiên bản

Cũng có thể xem trực tiếp trong app bằng cách bấm vào tag phiên bản cạnh logo. Mỗi báo cáo `vX.X` dưới đây có bản đầy đủ (sơ đồ kiến trúc, so sánh trước/sau, kết quả kiểm thử).

| Phiên bản | Tên | Nội dung chính |
| --- | --- | --- |
| `v3.8` | Giao diện đa theme | Thêm bộ chọn giao diện ngay trong app với 3 theme: `Tối · Midnight` (mặc định), `Sáng · Aurora` (nền trắng, sương màu pastel cam-tím-xanh-hồng trôi nhẹ liên tục, tự tắt animation khi hệ điều hành bật giảm chuyển động, card nổi bằng shadow) và `Rực rỡ · Sunset` (gradient hoàng hôn chuyển động nhẹ, card kính mờ/glass) — áp dụng ngay, tự nhớ qua `localStorage`. Bỏ anti-pattern gradient-text ở logo/heading, chuẩn hoá lại màu chữ/nền để đạt độ tương phản WCAG AA ở cả 3 theme, và sơ đồ tri thức AI (canvas) giờ đọc màu theo theme đang chọn thay vì hardcode cố định xanh-tím. ([báo cáo](BaoCao_MoodTune_v3.8.md)) |
| `v3.7` | Vận hành production ổn định hơn | Thêm app icon/favicon/PWA manifest và presence widget (số người online/đang nghe/tổng truy cập — số thật, đếm trực tiếp từ session); bật lại Audio Feature Engine (vá lỗi `tempo` array của `librosa` mới) và chuyển sang `waitress` thay Flask dev server; tự vá lệch Embedding/VOCAB sau restart; thêm rate-limit + admin key cho `/api/learn`. Không đổi kiến trúc model, tương thích ngược 100% với weights hiện có. ([báo cáo](BaoCao_MoodTune_v3.7.md)) |
| `v3.6` | Chuẩn hoá nhận diện phủ định & cụm từ cảm xúc | Sửa Rule Scorer: phủ định chưa được kiểm tra trong vòng lặp bigram, và `NEGATIONS` cũ chỉ nhận diện phủ định 1 từ (`không`, `chẳng`...) — các cụm 2-3 từ như `"không hề"`, `"chẳng bao giờ"` bị bỏ sót. Không đổi kiến trúc model, tương thích ngược 100% với weights hiện có. ([báo cáo](BaoCao_MoodTune_v3.6.md)) |
| `v3.5` | Chuẩn hoá theo Valence-Arousal | Rút từ 15 → 10 "cảm xúc đơn giản" để bám theo mô hình khoa học Valence-Arousal (GEMS/Circumplex); đổi tên nội bộ các cảm xúc sang tiếng Anh (`vui_ve` → `happy`,...). ([báo cáo](BaoCao_MoodTune_v3.5.md)) |
| `v3.1` | Thêm cảm xúc mới | Thêm 3 cảm xúc — Tự tin 💪, Biết ơn 🙏, Tức giận 😡 — nâng tổng số từ 12 lên 15 class. ([báo cáo](BaoCao_MoodTune_v3.1.md)) |
| `v3.0` | RLUF Bandit & Sơ đồ tri thức | Tự cài Thompson Sampling Multi-Armed Bandit (NumPy thuần, `bandit.py`) học gu nhạc từ Like/Dislike/Bỏ qua, cùng Sơ đồ tri thức cảm xúc tương tác (Canvas 2D) dựa trên trọng số attention của model. ([báo cáo](BaoCao_MoodTune_v3.0.md)) |
| `v2.5` | Adaptive Learning | Đổi ReLU → Leaky ReLU ở lớp ẩn (chống "chết nơ-ron" khi online learning liên tục) và thêm Adaptive L2 (hệ số weight decay tăng theo số lần feedback); thêm modal changelog trong app. ([báo cáo](BaoCao_MoodTune_v2.5.md)) |
| `v2.0` | Self-Attention | Thay Bag-of-Words bằng Embedding Layer + Self-Attention (Q,K,V) tự viết bằng NumPy, giữ đúng thứ tự từ (và phủ định) trong câu. Thêm Dynamic Vocab/Weight Expansion, Audio Feature Engine (BPM/Spectral Centroid/MFCC qua librosa) để re-rank nhạc, và Kho nhạc Local (Hybrid Online/Offline). ([báo cáo](BaoCao_MoodTune_v2.0.md)) |
| `v1.1` | Hoàn thiện giao diện | Hiển thị số phiên bản trên UI; hoàn thiện gợi ý cá nhân hoá, gợi ý theo giờ trong ngày, lịch sử phân tích. ([báo cáo](BaoCao_MoodTune_v1.1.md)) |
| `v1.0` | Phiên bản nền tảng | Rule Scorer + MLP Hybrid Engine (NumPy thuần, không dùng framework ML), tích hợp Jamendo API, feedback & online learning cơ bản. |

Hai tài liệu khác mô tả hệ thống theo chiều rộng thay vì theo từng phiên bản: `BaoCao_MoodTune_TongQuan.md` (kiến trúc tổng quan) và `BaoCao_MoodTune_DacTrung_AIEngine_API.md` (đặc trưng AI Engine & API).
