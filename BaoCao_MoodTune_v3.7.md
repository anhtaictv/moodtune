# MoodTune v3.7 — Báo cáo thay đổi so với v3.6

**Phiên bản:** `v3.7` (so với `v3.6` trong `BaoCao_MoodTune_v3.6.md`)
**Tên đầy đủ:** MoodTune — AI Cảm Xúc Tự Xây (RLUF Bandit · Knowledge Graph · Self-Attention) + Gợi ý nhạc Jamendo Hybrid
**Chủ đề nâng cấp:** Vận hành production ổn định hơn — bật lại Audio Feature Engine, chuyển sang WSGI server thật, vá lỗi đồng bộ vocab sau restart, chặn spam `/api/learn`, và thêm presence widget hiển thị số người dùng thật.

v3.7 là bản gộp 4 thay đổi vận hành/độ tin cậy được tích luỹ từ sau v3.6 — không đổi kiến trúc model, không cần retrain MLP, **tương thích ngược 100%** với `weights.npz` hiện có.

---

## Thay đổi 1: Bật lại Audio Feature Engine, chuyển sang waitress thay Flask dev server

### Mô tả
Tính năng **Audio Feature Engine** (BPM/Spectral Centroid/MFCC qua `librosa`, re-rank nhạc theo đặc trưng âm thanh — giới thiệu từ v2.0) thực chất **tắt từ đầu** vì thiếu thư viện `librosa` (chưa có wheel tương thích Python đang dùng). Khi cài được `librosa` và bật thử, lộ ra lỗi runtime: bản `librosa` mới trả `tempo` từ `beat_track()` dưới dạng `np.ndarray` (shape `(1,)`) thay vì Python scalar như các bản cũ — `float(tempo)` ném `TypeError: only 0-dimensional arrays can be converted to Python scalars`.

Đồng thời, backend vẫn chạy bằng Flask dev server (`app.run(...)`), tự in cảnh báo "WARNING: This is a development server. Do not use it in a production deployment" mỗi lần khởi động dù đã chạy qua pm2 ở production.

### Cài đặt kỹ thuật
- **`requirements.txt`**: thêm `librosa==0.11.0` và `waitress==3.0.2`.
- **`audio_features.py` → `analyze_track()`**: `bpm = float(tempo)` → `bpm = float(np.atleast_1d(tempo)[0])` — ép `tempo` về ít nhất 1 chiều trước khi lấy phần tử đầu, an toàn với cả hai dạng trả về (scalar cũ lẫn array mới).
- **`app.py` → entrypoint**: thay `app.run(host="0.0.0.0", port=5005, debug=False)` bằng `from waitress import serve; serve(app, host="0.0.0.0", port=5005, threads=1)`. Giữ `threads=1` chủ động: `AttentionMLP`/`EmotionEngine` là NumPy thuần tự viết, mutate trực tiếp `self.E`/`self._ids`/... không có lock — 2 request `/api/learn` xử lý đồng thời trên nhiều thread sẽ ghi đè state lẫn nhau và làm hỏng model chung. `threads=1` giữ đúng hành vi tuần tự như dev server cũ, chỉ đổi server lõi (WSGI thật, không cảnh báo, ổn định hơn dưới production load read-heavy của `/api/predict`).
- **`.gitignore`**: thêm `backend/backups/` (snapshot trước mỗi lần `gemini_teacher.py` dạy hàng loạt) và `backend/audio_cache.json` (cache đặc trưng audio tự sinh lại được) — tránh commit dữ liệu sinh ra lúc runtime.

### Kết quả kiểm thử
Tái hiện trực tiếp lỗi cũ và xác nhận fix, dùng tín hiệu có beat thật (click track 120 BPM) để `beat_track()` trả về đúng dạng array mới:

```
tempo array: [117.45383523] (1,)
float(tempo)                        -> FAILS: TypeError('only 0-dimensional arrays can be converted to Python scalars')
float(np.atleast_1d(tempo)[0])      -> 117.45383522727273   (OK)
```

→ Xác nhận đúng nguyên nhân lỗi mô tả trong commit và fix giải quyết triệt để, không phụ thuộc việc `librosa` trả scalar hay array.

---

## Thay đổi 2: Tự vá lệch Embedding/VOCAB sau restart

### Mô tả
Khi chạy `gemini_teacher.py` dạy model hàng loạt rồi restart service, có trường hợp `weights.npz` (chứa ma trận Embedding `E`) được nạp lại **cũ hơn** `dynamic_vocab.json`/`dynamic_lexicon.json` đã ghi xuống đĩa (lệch thời điểm save giữa 2 loại file trạng thái). Hệ quả: `E.shape[0] < VOCAB_SIZE` hiện tại — `forward()` vẫn chạy được vì chỉ index những token có trong câu, nhưng `backward()` xử lý gradient theo toàn bộ `VOCAB_SIZE` thì ăn `IndexError` ngay khi gặp token có id ≥ `E.shape[0]`.

### Cài đặt kỹ thuật
- **`emotion_mlp.py` → `AttentionMLP.forward()`**: thêm kiểm tra đầu hàm, ngay sau early-return cho input rỗng — nếu `self.E.shape[0] < VOCAB_SIZE`, gọi `self.expand_vocab(VOCAB_SIZE - self.E.shape[0])` để vstack thêm dòng cho khớp, trước khi tính `X = self.E[token_ids]`. Tự vá ngay tại điểm phát sinh lệch, không cần sửa logic load/save.

### Kết quả kiểm thử
Mô phỏng đúng tình huống thực tế: tạo `AttentionMLP`, rồi cắt bớt `E` cho nhỏ hơn `VOCAB_SIZE` (giả lập nạp `weights.npz` cũ), gọi `forward()` với token id chạm tới vùng vừa bị cắt:

```
[Init] Vocab=4201 (+2573 dynamic word, +1043 dynamic phrase) | Classes=10
initial E rows: 4201   VOCAB_SIZE: 4201
shrunk E rows:  4196                      # giả lập weights.npz cũ hơn vocab
forward([0, 1, 4200]) -> OK, không IndexError
E rows sau self-heal: 4201                # tự vá khớp lại VOCAB_SIZE
output shape: (10,)
```

→ Xác nhận `forward()` tự phát hiện và vá lệch ngay trong lần gọi đầu tiên sau restart, không cần can thiệp thủ công hay mất dữ liệu đã học.

---

## Thay đổi 3: Rate-limit + Admin Key cho `/api/learn`

### Mô tả
`/api/learn` ghi trực tiếp vào model online learning **chung cho mọi người dùng** (không phải instance riêng per-user) — 1 IP gửi spam nhãn sai liên tục có thể đầu độc (poisoning) model dùng chung cho tất cả. Trước v3.7, endpoint này không có giới hạn nào.

### Cài đặt kỹ thuật
- **`app.py`**: thêm `_client_ip()` (đọc `X-Forwarded-For` nếu có, fallback `request.remote_addr`), `_is_rate_limited(ip)` (sliding window in-memory: tối đa `LEARN_RATE_LIMIT=12` request/`LEARN_RATE_WINDOW=60` giây mỗi IP), và `_is_admin_request()` (so khớp header `X-Admin-Key` với env `MOODTUNE_ADMIN_KEY`).
- **`/api/learn`**: chặn đầu route — nếu không phải admin request và đã vượt rate limit, trả `429` kèm message tiếng Việt giải thích giới hạn. Script dạy AI nội bộ (`gemini_teacher.py`) gửi header `X-Admin-Key` để bỏ qua giới hạn này; nếu không set `MOODTUNE_ADMIN_KEY` thì bypass tắt hẳn (không có giá trị nào khớp được với rỗng).
- **`ecosystem.config.js`**: pass-through `MOODTUNE_ADMIN_KEY` từ biến môi trường của host (không hardcode secret vào file commit).

### Kết quả kiểm thử
Test trực tiếp logic sliding-window (tách khỏi Flask để kiểm chứng đúng phần đếm):

```
15 request liên tiếp cùng 1 IP trong < 60s:
[False, False, False, False, False, False, False, False, False, False, False, False, True, True, True]
                                                                                ^^^^
request thứ 13 trở đi bị chặn (limit=12) -> khớp đúng thiết kế "12 request/60s/IP"
```

---

## Thay đổi 4: Presence widget + PWA icon/manifest

### Mô tả
Trước v3.7, MoodTune chưa có icon chuẩn khi thêm vào màn hình chính điện thoại (thiếu favicon/apple-touch-icon/manifest → hiển thị viền trắng/đen), và chưa có cách nào cho người dùng thấy "có người khác đang dùng app". Bổ sung bộ icon đầy đủ (`favicon.png`, `apple-touch-icon.png`, `icon-192.png`, `icon-512.png`, `manifest.json` chuẩn PWA `display: standalone`), logo cạnh tên app ở header, và **presence widget** (góc dưới phải UI) hiển thị số người đang online/đang nghe/tổng lượt truy cập — toàn bộ là **số thật**, đếm trực tiếp từ session đang hoạt động, không làm tròn/giả lập gì thêm.

### Cài đặt kỹ thuật
- **PWA/icon**: thêm các thẻ `<link rel="icon">`, `<link rel="manifest">`, `<link rel="apple-touch-icon">` và các meta `theme-color`/`apple-mobile-web-app-*` vào `<head>`; `manifest.json` mới khai báo icon 192/512, `display: standalone`.
- **Presence widget — backend**: `online_sessions` (dict `session_id -> {last_seen, listening}` trong RAM, dọn định kỳ bằng `_prune_sessions()` theo `PRESENCE_TIMEOUT=40s`) và route `POST /api/presence/ping`, trả về `online = len(online_sessions)`, `listening = số session có listening=True`. Tổng lượt truy cập (`visit_total`, persist ở `visit_stats.json`) tăng +1 mỗi `session_id` mới gặp lần đầu (`counted_sessions`), không tăng lại khi cùng session ping tiếp.
- **Presence widget — frontend**: sinh `SESSION_ID` ổn định theo tab (`sessionStorage`), gọi `pingPresence()` mỗi `PRESENCE_PING_MS=15000`ms và mỗi lần play/pause, kèm `sendBeacon` báo rời trang khi đóng/chuyển tab để server không phải đợi hết `PRESENCE_TIMEOUT`; cập nhật 3 số trong `.presence-widget`.

> *Ghi chú:* bản nháp giữa kỳ của thay đổi này có thử thêm một baseline ngẫu nhiên cộng vào số online/listening để demo cho có số liệu lúc viết báo cáo — đã bỏ hẳn trước khi đưa vào v3.7, vì widget hiển thị cho người dùng thật thì số phải là số thật.

---

## Bảng so sánh tổng quan v3.6 vs v3.7

| Khía cạnh | v3.6 | v3.7 |
|---|---|---|
| Audio Feature Engine (BPM/Spectral/MFCC re-rank) | Tắt (thiếu `librosa`, lỗi `float(tempo)` nếu bật) | **Bật, hoạt động đúng** với cả `librosa` bản mới |
| WSGI server production | Flask dev server (`app.run`, tự cảnh báo) | **`waitress`** (`threads=1`, không cảnh báo) |
| Đồng bộ Embedding `E` / `VOCAB_SIZE` sau restart | Có thể lệch → `IndexError` ở `backward()` | **Tự vá** trong `forward()`, không cần can thiệp |
| `/api/learn` chống spam/poisoning | Không giới hạn | **12 request/60s/IP**, bypass qua `X-Admin-Key` |
| PWA icon/manifest | Không có (favicon mặc định) | **Có** — favicon/apple-touch-icon/manifest.json chuẩn, thêm vào màn hình chính không bị viền |
| Bộ đếm online/listening/tổng truy cập | Không có | **Presence widget** — số thật, đếm trực tiếp từ session đang hoạt động |
| Tương thích `weights.npz` cũ | — | **100%** (không đổi kiến trúc, không cần retrain) |
| Kiến trúc MLP / số class | Không đổi | Không đổi |
| Version trên UI | `v3.6` | **`v3.7`** + entry Changelog mới |

---

## Triển khai

- Commit liên quan: `e6f834a` (PWA icon/manifest + presence widget), `9ac27e1` (vá đồng bộ Embedding/VOCAB + rate-limit/admin key), `314b15e` (bật Audio Feature Engine, chuyển sang waitress), cộng bản dọn bỏ baseline giả lập khỏi presence widget trước khi phát hành v3.7.
- Production (`pm2`, process `moodtune-backend`) đã chạy bản `314b15e` này được hơn 1 giờ tại thời điểm viết báo cáo, `status: online`.
- `GET /api/health` → `200 OK`, kiến trúc không đổi (`vocab_size: 4201`, `output_size: 10`, `feedback_count: 3551`) — xác nhận các thay đổi không phá vỡ trạng thái model đang học online.

> **Định hướng tiếp theo:** Rate-limit hiện lưu in-memory (mất khi restart process) — nếu cần chính xác hơn qua các lần restart hoặc scale nhiều worker, sẽ cần chuyển sang lưu chung (Redis/file) thay vì biến trong tiến trình.
