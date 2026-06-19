# MoodTune v3.6 — Báo cáo thay đổi so với v3.5

**Phiên bản:** `v3.6` (so với `v3.5` trong `BaoCao_MoodTune_v3.5.md`)
**Tên đầy đủ:** MoodTune — AI Cảm Xúc Tự Xây (RLUF Bandit · Knowledge Graph · Self-Attention) + Gợi ý nhạc Jamendo Hybrid
**Chủ đề nâng cấp:** Chuẩn hoá khả năng nhận diện tiếng Việt của Rule Scorer — sửa lỗi phủ định (negation) và lỗ hổng học cụm từ cảm xúc mới (dynamic vocab)

v3.6 là bản vá tập trung vào **2 lỗi nhận diện** được phát hiện khi rà soát lại `rule_score()` và cơ chế mở rộng vocab động — không đổi kiến trúc model, không cần retrain MLP, **tương thích ngược 100%** với `weights.npz` hiện có.

---

## Thay đổi 1: Sửa lỗi phủ định (Negation) trong Rule Scorer

### Mô tả
Rà soát `rule_score()` (`emotion_mlp.py`) phát hiện 2 lỗi khiến phủ định tiếng Việt không được nhận diện đúng:

1. **Negation chỉ áp dụng cho unigram.** Vòng lặp bigram quét cụm 2 từ (`"tập trung"`, `"chữa lành"`, `"ngày xưa"`...) hoàn toàn không kiểm tra từ phủ định đứng trước. Ví dụ câu *"tôi không tập trung được"* vẫn cộng điểm dương cho `focused` (do bigram `"tập trung"` ăn điểm ×1.5 mà không bị đảo dấu), dẫn đến kết quả sai hoàn toàn so với ý nghĩa câu.
2. **Chỉ nhận diện phủ định 1 từ.** `NEGATIONS` cũ chỉ có các từ đơn (`không`, `chẳng`, `chả`...) và code chỉ so khớp đúng `words[i-1]`. Phủ định 2-3 từ rất phổ biến trong tiếng Việt như *"không hề"*, *"chẳng bao giờ"*, *"chưa bao giờ"* **không được nhận diện** — ví dụ *"tôi không hề vui"* bị tính y như *"tôi vui"` vì từ ngay trước `"vui"` là `"hề"`, không nằm trong tập phủ định.

### Cài đặt kỹ thuật
- **`lexicon.py` → `NEGATIONS`**: Từ `set` 8 từ đơn mở rộng thành tập gồm cả cụm 2-3 từ:

| Độ dài | Các cụm thêm mới |
|---|---|
| 1 từ (giữ nguyên) | `không`, `chẳng`, `chả`, `đâu`, `chưa`, `khỏi`, `ko`, `k` |
| 2 từ (mới) | `không hề`, `chẳng hề`, `chả hề`, `chưa hề`, `không phải`, `chẳng phải`, `chả phải`, `không có`, `đâu có`, `có đâu`, `đâu phải` |
| 3 từ (mới) | `không bao giờ`, `chẳng bao giờ`, `chả bao giờ`, `chưa bao giờ` |

- **`emotion_mlp.py` → hàm mới `_is_negated(words, idx)`**: thay cho so khớp `words[i-1] in NEGATIONS` cũ. Hàm quét các cụm con dài 1..`NEGATION_MAX_LEN` (tính tự động từ độ dài cụm dài nhất trong `NEGATIONS`, hiện = 3) ngay trước vị trí `idx`, trả `True` nếu khớp bất kỳ cụm phủ định nào.
- **`rule_score()`**: gọi `_is_negated()` ở **cả vòng lặp bigram và unigram** (trước đây bigram không có biến `negated` nào cả).
- Không cần sửa gì ở `to_token_ids()`/`_tokenize()` — đây là nhánh riêng cho MLP, MLP tự học phủ định qua attention từ dữ liệu feedback, không phụ thuộc `NEGATIONS`.

### Kết quả kiểm thử
Gọi trực tiếp `rule_score()` (tách khỏi MLP để kiểm chứng đúng phần rule):

| Câu input | Trước fix (suy luận từ code cũ) | Sau fix |
|---|---|---|
| `"hôm nay tôi rất vui"` | `happy` (đúng) | `happy` (đúng, không đổi) |
| `"tôi không hề vui"` | `happy` (**sai** — không bắt được phủ định 2 từ) | điểm `happy` bị triệt tiêu → uniform (đúng) |
| `"tôi chẳng bao giờ vui khi ở đây"` | `happy` (**sai** — không bắt được phủ định 3 từ) | điểm `happy` bị triệt tiêu → uniform (đúng) |
| `"tôi không tập trung được"` | `focused` (**sai** — bigram không bị phủ định) | điểm `focused` bị triệt tiêu → uniform (đúng) |
| `"cô ấy không bao giờ tập trung"` | `focused` (**sai**) | điểm `focused` bị triệt tiêu → uniform (đúng) |

> *Lưu ý:* kết quả "uniform" (đều nhau, không lệch về emotion nào) là hành vi đúng theo thiết kế hiện tại — phủ định nhân điểm dương duy nhất thành âm, sau đó bị `np.maximum(scores, 0)` chặn về 0, dẫn đến không match emotion nào ở tầng rule (model Hybrid vẫn có tầng MLP bổ trợ phần này).

---

## Thay đổi 2: Dynamic Lexicon Expansion — học cụm từ cảm xúc mới

### Mô tả
Cơ chế **Dynamic Weight Expansion** (từ v2.0) khi có feedback chứa từ hoàn toàn mới (OOV) chỉ làm 2 việc: thêm từ đơn vào `VOCAB`/`VOCAB_IDX` và mở rộng ma trận Embedding `E`. Hệ quả: **chỉ phần MLP học được từ mới**, còn **Rule Scorer (`rule_score()`) hoàn toàn không biết gì** vì nó chỉ so khớp với `LEXICON` — một từ điển tĩnh không được cập nhật lúc runtime. Cụm cảm xúc mới (bigram) do người dùng dạy qua feedback do đó bị "mất" ở tầng rule, dù model đã "học" ở tầng MLP.

### Cài đặt kỹ thuật
- **File mới `dynamic_lexicon.json`** (cùng cấp với `dynamic_vocab.json`): lưu cụm từ học được, dạng `{"emotion": {"cụm từ": weight}}`.
- **`emotion_mlp.py` → `find_new_phrases(text, oov_words)`**: với mỗi từ OOV vừa phát hiện, tìm các bigram trong câu có chứa từ đó (ngữ cảnh 2 từ xung quanh từ mới) → đây là ứng viên cụm cảm xúc cần học.
- **`emotion_mlp.py` → `add_lexicon_phrases(phrases, emotion, weight=2.0)`**: ghi cụm mới vào `LEXICON[emotion]` (rule scorer dùng được ngay lập tức, không cần restart) **và** vào `VOCAB`/`VOCAB_IDX` (để `_tokenize()` ưu tiên match cụm này thành 1 token cho MLP), đồng thời persist xuống `dynamic_lexicon.json`. Weight mặc định `2.0` — mức "liên quan rõ" theo thang đã quy ước trong `lexicon.py` (1.0–1.5 yếu, 2.0–2.5 rõ, 3.0 mạnh nhất).
- **Khởi động module**: load + merge `dynamic_lexicon.json` vào `LEXICON` và `VOCAB` ngay khi import, theo đúng thứ tự đã lưu — đảm bảo số dòng ma trận Embedding `E` khi load lại từ `weights.npz` luôn khớp với `VOCAB_SIZE` tái dựng (không bị lệch index sau restart).
- **`EmotionEngine.learn()`**: gọi `find_new_phrases()` + `add_lexicon_phrases()` ngay sau `add_vocab_words()`, cộng tổng số ô embedding mới (từ đơn + cụm) rồi gọi `expand_vocab()` một lần duy nhất — giữ đúng thứ tự tăng trưởng vocab để `np.vstack()` không làm lệch dòng.
- Nếu cụm sinh ra đã tồn tại sẵn trong `LEXICON` (ví dụ từ mới nằm trong một bigram tĩnh có sẵn như `"chữa lành"`), hàm bỏ qua không ghi đè — tránh trùng lặp/đè weight gốc.

### Kết quả kiểm thử
Kiểm thử end-to-end `learn()` trên bản sao cách ly (không đụng dữ liệu production):

```
Input: "nhạc này nghe thật chữa lành tâm hồn", correct_emotion="relaxed"

[Vocab] +8 từ mới, +6 cụm mới -> vocab_size=603
LEXICON['relaxed']['thật chữa'] = 2.0   # cụm mới được học
vocab_size: 589 -> 603  |  E.shape: (589, 32) -> (603, 32)   # khớp nhau, không lệch dòng
```

Restart engine từ file đã lưu (kiểm tra persistence không bị mất dữ liệu/lệch index):

```
[Init] Vocab=603 (+8 dynamic word, +6 dynamic phrase) | Classes=10
LEXICON['relaxed'] có "thật chữa": True
predict("nhạc này nghe thật chữa lành tâm hồn") -> relaxed, 100.0%
```

→ Cụm cảm xúc mới được rule scorer nhận diện ngay từ lần dự đoán kế tiếp, không chỉ riêng MLP, và không mất dữ liệu qua restart.

---

## Bảng so sánh tổng quan v3.5 vs v3.6

| Khía cạnh | v3.5 | v3.6 |
|---|---|---|
| `NEGATIONS` | 8 từ đơn | **8 từ đơn + 11 cụm 2 từ + 4 cụm 3 từ** |
| Phủ định áp dụng cho | Chỉ unigram | **Unigram lẫn bigram** |
| Phủ định nhiều từ ("không hề", "chẳng bao giờ") | Không nhận diện được | **Nhận diện được** (quét cụm con 1..3 từ) |
| Dynamic Weight Expansion học được | Chỉ từ đơn (OOV) → MLP embedding | **Từ đơn (MLP) + cụm bigram (LEXICON + MLP)** |
| File lưu trạng thái động | `dynamic_vocab.json` | **+ `dynamic_lexicon.json`** (mới) |
| Tương thích `weights.npz` cũ | — | **100%** (không đổi kiến trúc, không cần retrain) |
| Kiến trúc MLP / số class | Không đổi | Không đổi |
| Version trên UI | `v3.5` | **`v3.6`** + entry Changelog mới |

---

## Triển khai

- Commit: `v3.6: chuẩn hoá nhận diện phủ định & cụm từ cảm xúc tiếng Việt` (file thay đổi: `backend/lexicon.py`, `backend/emotion_mlp.py`, `frontend/index.html`).
- Deploy production: `pm2 restart moodtune-backend` — khởi động lại sạch, không lỗi, `feedback_count`/`vocab_size`/`replay` giữ nguyên trạng thái học cũ (`feedback=2042, vocab=2579`).
- `GET /api/health` sau deploy → `200 OK`, kiến trúc không đổi (`vocab_size: 2579`, `output_size: 10`) — xác nhận bản vá không phá vỡ trạng thái model đang chạy.

> **Định hướng tiếp theo:** Cân nhắc thêm preprocessing chuẩn hoá văn bản không dấu (ví dụ "khong he vui") về có dấu trước khi qua `rule_score()`, vì hiện tại `NEGATIONS`/`LEXICON` chỉ so khớp chính xác theo dạng có dấu Unicode NFC.
