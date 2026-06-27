"""
MoodTune Emotion Engine - Hybrid AI Model (v4.0 - Word Segmentation / Model 3)
============================================================================
Kiến trúc 2 tầng:

1. RULE SCORER:
   - Từ điển cảm xúc với weighted scoring (nhanh, chính xác với từ khoá)
   - N-gram boost ×1.5 (cụm 2..MAX_LEXICON_PHRASE_LEN từ ăn điểm gấp 1.5 —
     tổng quát hoá từ bigram-only trước v4.0, xem rule_score())
   - Negation handling: từ/cụm phủ định 1-3 từ ("không", "không hề",
     "chẳng bao giờ"...) đứng ngay trước unigram HOẶC cụm dài hơn → đảo dấu
     điểm (×-0.6)

2. ATTENTION MLP LEARNER (numpy thuần, không dùng framework):
   - Word Segmentation longest-match (v4.0): _tokenize() ghép cụm DÀI NHẤT
     có sẵn trong VOCAB_IDX trước khi rơi về từ đơn, không còn giới hạn 2 từ
     như bản cũ — sửa "vocab chết" (cụm cảm xúc 3-5 từ trong LEXICON có
     embedding nhưng tokenizer cũ không bao giờ tạo ra được token đó).
   - NEGATIONS được đăng ký thẳng vào VOCAB từ đầu (v4.0) — Attention có cơ
     hội tự học pattern phủ định qua dữ liệu, không chỉ phụ thuộc rule cứng.
   - Embedding(VOCAB_SIZE, d=32) -> Self-Attention(Q,K,V) -> mean-pool
     -> Dense(d->64, Leaky ReLU) -> Dense(64->10, Softmax)
   - Attention(Q,K,V) = Softmax(Q K^T / sqrt(d_k)) V, forward+backward thủ công
   - Leaky ReLU (max(0.01x, x)) ở lớp ẩn -> tránh "chết nơ-ron" (Dying ReLU)
   - SGD + Momentum 0.9 + Adaptive L2 regularization (weight decay tăng dần
     theo số lần feedback, chặn ở 5x giá trị gốc 1e-4)
   - He initialization
   - Online learning từ feedback + experience replay (≤500 mẫu)
     để tránh catastrophic forgetting
   - Dynamic Weight Expansion: từ mới hoàn toàn -> mở rộng Embedding Matrix E
     bằng np.vstack() (He init x 0.01) ngay tại runtime, không cần restart.
     Cụm cảm xúc mới (bigram chứa từ đó) cũng được thêm vào LEXICON
     (dynamic_lexicon.json) để rule scorer học được, không chỉ MLP embedding

3. HYBRID BLEND:
   - final = alpha * rule + (1-alpha) * mlp
   - alpha bắt đầu = 0.85, giảm theo công thức:
       alpha = max(0.35, 0.85 - 0.5 * fc / (fc + 40))
   - Càng nhiều feedback → càng tin MLP (alpha giảm về 0.35)
"""

import numpy as np
import json
import os
import random
import re
import sys
import unicodedata

# Ép stdout/stderr dùng UTF-8 để print tiếng Việt không crash trên Windows (cp1252)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ─── IMPORT TỪ ĐIỂN ─────────────────────────────────────────────
# Mọi thứ liên quan đến từ điển nằm trong lexicon.py
# Chỉ chỉnh lexicon.py khi muốn thêm/bớt từ hoặc class cảm xúc.
from lexicon import EMOTIONS, EMOTION_META, LEXICON, NEGATIONS

N = len(EMOTIONS)

# ─── VOCAB (tĩnh từ lexicon + động từ dynamic_vocab.json) ───────
VOCAB = []
VOCAB_IDX = {}
for emo, words in LEXICON.items():
    for w in words:
        if w not in VOCAB_IDX:
            VOCAB_IDX[w] = len(VOCAB)
            VOCAB.append(w)

# Đăng ký NEGATIONS như "function token" (v4.0 - Model 3): trước đây các từ
# phủ định ("không", "chẳng hề", "không bao giờ"...) chỉ vào được VOCAB một
# cách tình cờ NẾU từng xuất hiện gần một từ cảm xúc trong feedback (qua
# add_vocab_words() trong learn()) - không đảm bảo đầy đủ, và embedding học
# được không mang ngữ nghĩa "đây là token phủ định" nhất quán. Đăng ký thẳng
# từ đầu để Attention luôn có cơ hội tự học pattern phủ định qua dữ liệu,
# không chỉ phụ thuộc 100% vào rule_score()/_is_negated().
for w in NEGATIONS:
    if w not in VOCAB_IDX:
        VOCAB_IDX[w] = len(VOCAB)
        VOCAB.append(w)

DYNAMIC_VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_vocab.json")
DYNAMIC_LEXICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_lexicon.json")


def _load_dynamic_vocab():
    if os.path.exists(DYNAMIC_VOCAB_PATH):
        try:
            with open(DYNAMIC_VOCAB_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _load_dynamic_lexicon():
    if os.path.exists(DYNAMIC_LEXICON_PATH):
        try:
            with open(DYNAMIC_LEXICON_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


DYNAMIC_VOCAB = _load_dynamic_vocab()
for _w in DYNAMIC_VOCAB:
    if _w not in VOCAB_IDX:
        VOCAB_IDX[_w] = len(VOCAB)
        VOCAB.append(_w)

# Cụm cảm xúc (bigram) học được từ feedback — xem add_lexicon_phrases().
# Merge thẳng vào LEXICON để rule_score() nhận diện được ngay khi khởi
# động, không chỉ riêng MLP embedding mới "biết" cụm từ này.
DYNAMIC_LEXICON = _load_dynamic_lexicon()
for _emo, _phrases in DYNAMIC_LEXICON.items():
    for _phrase, _weight in _phrases.items():
        LEXICON.setdefault(_emo, {})[_phrase] = _weight
        if _phrase not in VOCAB_IDX:
            VOCAB_IDX[_phrase] = len(VOCAB)
            VOCAB.append(_phrase)

VOCAB_SIZE = len(VOCAB)
_n_dynamic_phrases = sum(len(p) for p in DYNAMIC_LEXICON.values())
print(f"[Init] Vocab={VOCAB_SIZE} (+{len(DYNAMIC_VOCAB)} dynamic word, "
      f"+{_n_dynamic_phrases} dynamic phrase) | Classes={N}")

# Độ dài cụm dài nhất hiện có trong VOCAB (tính theo số từ) — dùng cho
# longest-match segmentation ở _tokenize() (v4.0 - Model 3). Trước đây
# _tokenize() chỉ thử ghép đúng 2 từ (bigram-only) nên các cụm cảm xúc dài
# hơn trong LEXICON (vd 5 từ "không thể chấp nhận được") không bao giờ được
# nhận diện thành 1 token - "vocab chết", có embedding nhưng chưa từng được
# forward/backward chạm tới. Cụm mới học runtime qua add_lexicon_phrases()
# luôn ≤2 từ nên không cần tính lại giá trị này sau khi module đã load.
MAX_PHRASE_LEN = max((len(w.split()) for w in VOCAB if " " in w), default=1)


# ─── EMOJI → TOKEN ───────────────────────────────────────────────
# Thay emoji bằng token tiếng Việt trước khi regex strip — không thì emoji
# bị xoá im lặng dù là tín hiệu cảm xúc rất mạnh trong text online.
EMOJI_MAP = {
    # Happy
    "😄":"vui","😊":"vui","😀":"vui","😃":"vui","😁":"vui","😆":"vui",
    "😂":"vui","🥳":"vui","🎉":"vui","🤩":"vui","🙂":"vui","☺":"vui",
    # Sad
    "😢":"buồn","😭":"khóc","💔":"đau lòng","😞":"buồn",
    "😔":"buồn","😟":"buồn","🥺":"tủi","😣":"khổ","😿":"buồn",
    # Romantic
    "🥰":"yêu","😍":"si mê","💕":"yêu thương","❤":"yêu",
    "💖":"yêu","💗":"yêu","💓":"yêu","😘":"hôn","💝":"yêu",
    # Energetic
    "⚡":"năng động","🔥":"hype","💪":"workout","🏃":"chạy",
    "🎊":"party","🕺":"nhảy","💃":"nhảy",
    # Relaxed
    "😌":"bình yên","🌿":"bình yên","☕":"cà phê",
    "🍃":"nhẹ nhàng","🌸":"dịu dàng","🫖":"trà","😴":"ngủ",
    # Lonely
    "🌙":"cô đơn","🌑":"cô đơn","😶":"im lặng","🥀":"buồn",
    # Stressed
    "😰":"lo lắng","😫":"mệt mỏi","😩":"kiệt sức",
    "😓":"căng thẳng","🤯":"quá tải","😤":"bực mình",
    # Focused
    "🎯":"tập trung","📚":"học","💻":"code","📝":"viết","🧠":"deep work",
    # Nostalgic
    "🍂":"hoài niệm","📷":"kỷ niệm","🕰":"ngày xưa","📸":"kỷ niệm",
    # Angry
    "😡":"tức giận","🤬":"điên tiết","💢":"giận dữ","😠":"bực tức",
}


def _expand_emojis(text):
    for emoji, token in EMOJI_MAP.items():
        text = text.replace(emoji, f" {token} ")
    return text


# ─── TIỀN XỬ LÝ ─────────────────────────────────────────────────
def preprocess(text):
    text = _expand_emojis(text)
    # Chuẩn hoá NFC: nếu input dùng Unicode tổ hợp dấu rời (NFD — hay gặp khi
    # copy từ một số nguồn/app khác nhau), dấu câu sẽ là ký tự combining mark
    # riêng (Mn) và bị regex dưới đây xoá mất vì whitelist chỉ liệt kê ký tự
    # có dấu dạng dựng sẵn (NFC) -> không chuẩn hoá sẽ làm vỡ từ ("việt" -> "vi").
    t = unicodedata.normalize("NFC", text).lower().strip()
    # Giữ dấu tiếng Việt
    t = re.sub(r'[^\w\sàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹ]', ' ', t)
    return t


# ─── RULE SCORER ────────────────────────────────────────────────
# NEGATIONS (lexicon.py) giờ chứa cả cụm phủ định nhiều từ ("không hề",
# "chẳng bao giờ"). NEGATION_MAX_LEN = số từ dài nhất trong các cụm đó,
# dùng để giới hạn cửa sổ quét lùi trong _is_negated().
NEGATION_MAX_LEN = max(len(p.split()) for p in NEGATIONS)

# Phủ định tu từ: "X gì chứ/đâu/vậy/nữa" — người nói phủ nhận cảm xúc bằng
# câu hỏi tu từ thay vì "không X" tường minh. Rule scorer cần bắt pattern này
# riêng vì _is_negated() chỉ quét về phía TRÁI (prefix negation), còn phủ
# định tu từ lại nằm về phía PHẢI của từ cảm xúc.
RHETORICAL_NEG_MARKERS = frozenset({"chứ", "đâu", "vậy", "nữa"})


def _is_rhetorically_negated(words, idx, length=1):
    """True nếu cụm words[idx:idx+length] bị phủ định bởi pattern
    '(cụm) + gì + (chứ|đâu|vậy|nữa)' ngay sau đó."""
    end = idx + length
    if end < len(words) and words[end] == "gì":
        return end + 1 >= len(words) or words[end + 1] in RHETORICAL_NEG_MARKERS
    return False


# Độ dài cụm dài nhất trong LEXICON (v4.0 - Model 3): trước đây rule_score()
# chỉ thử ghép đúng 2 từ liền kề (bigram-only) nên các cụm cảm xúc dài hơn
# (vd 5 từ "không thể chấp nhận được" trong LEXICON["angry"]) không bao giờ
# khớp được - dù tác giả lexicon đã cố tình viết cụm đó. ~9% (54/591) entry
# trong LEXICON dài hơn 2 từ và đều rơi vào tình trạng này trước v4.0.
MAX_LEXICON_PHRASE_LEN = max(
    (len(p.split()) for words in LEXICON.values() for p in words if " " in p),
    default=2,
)


def _is_negated(words, idx):
    """True nếu từ/cụm bắt đầu ở vị trí idx bị phủ định bởi 1..N từ đứng
    ngay liền trước (N = NEGATION_MAX_LEN). Quét các cụm con ngay trước
    idx từ ngắn (1 từ) đến dài để bắt được cả phủ định 1 từ ("không vui")
    và phủ định nhiều từ ("không hề vui", "chẳng bao giờ vui")."""
    for length in range(1, min(NEGATION_MAX_LEN, idx) + 1):
        if " ".join(words[idx-length:idx]) in NEGATIONS:
            return True
    return False


def rule_score(text):
    t = preprocess(text)
    scores = np.zeros(N)
    words = t.split()

    # N-gram matching (cụm 2..MAX_LEXICON_PHRASE_LEN từ) — tổng quát hoá từ
    # bigram-only (trước v4.0). Cũng bị phủ định nếu có từ/cụm phủ định đứng
    # ngay trước cụm (từ v3.6: áp dụng phủ định cho cả cụm, không chỉ unigram)
    for length in range(2, MAX_LEXICON_PHRASE_LEN + 1):
        for i in range(len(words) - length + 1):
            phrase = " ".join(words[i:i+length])
            negated = _is_negated(words, i) or _is_rhetorically_negated(words, i, length)
            for ei, emo in enumerate(EMOTIONS):
                if phrase in LEXICON[emo]:
                    val = LEXICON[emo][phrase] * 1.5  # phrase boost (giữ hệ số bigram cũ)
                    scores[ei] += (-0.6 * val) if negated else val

    # Unigram + xử lý phủ định (prefix và tu từ "X gì chứ")
    for i, w in enumerate(words):
        negated = _is_negated(words, i) or _is_rhetorically_negated(words, i)
        for ei, emo in enumerate(EMOTIONS):
            if w in LEXICON[emo]:
                val = LEXICON[emo][w]
                # phủ định → giảm mạnh điểm cảm xúc đó thay vì cộng
                scores[ei] += (-0.6 * val) if negated else val

    scores = np.maximum(scores, 0)   # không cho điểm âm
    # Softmax
    if scores.max() == 0:
        return np.ones(N) / N  # uniform nếu không match
    e = np.exp(scores - scores.max())
    return e / e.sum()


# ─── TOKEN SEQUENCE CHO ATTENTION MLP (Ý tưởng 3 + Word Segmentation v4.0) ─
def _tokenize(text, max_len=24):
    """Chuyển câu -> danh sách (token_str, token_id_or_None), GIỮ THỨ TỰ từ
    trong câu (khác Bag-of-Words vốn mất thông tin trật tự).

    Word segmentation bằng longest-match (v4.0 - Model 3): tại mỗi vị trí,
    thử ghép cụm DÀI NHẤT có sẵn trong VOCAB_IDX trước (từ MAX_PHRASE_LEN từ
    giảm dần xuống 2), rồi mới rơi về từ đơn. Tổng quát hoá so với bản cũ
    (chỉ thử đúng 2 từ/bigram) - sửa được các cụm cảm xúc 3-5 từ trong
    LEXICON (vd "không thể chấp nhận được") mà tokenizer cũ không bao giờ
    chạm tới dù embedding của chúng đã tồn tại trong VOCAB từ đầu.

    Token OOV (chưa có trong VOCAB_IDX) vẫn được giữ lại với id=None để
    phục vụ Knowledge Graph (Ý tưởng 3, nangcap2.txt)."""
    t = preprocess(text)
    words = t.split()
    tokens = []
    i = 0
    while i < len(words) and len(tokens) < max_len:
        matched = False
        max_try = min(MAX_PHRASE_LEN, len(words) - i)
        for length in range(max_try, 1, -1):
            phrase = " ".join(words[i:i+length])
            if phrase in VOCAB_IDX:
                tokens.append((phrase, VOCAB_IDX[phrase]))
                i += length
                matched = True
                break
        if not matched:
            w = words[i]
            tokens.append((w, VOCAB_IDX.get(w)))
            i += 1
    return tokens


def to_token_ids(text, max_len=24):
    """Chuyển câu -> danh sách index trong VOCAB, bỏ qua từ OOV."""
    return [tid for _, tid in _tokenize(text, max_len) if tid is not None]


# ─── DYNAMIC VOCAB EXPANSION (Ý tưởng 1) ─────────────────────────
def find_oov_words(text):
    """Tìm các từ đơn (>=2 ký tự) trong câu chưa có trong VOCAB_IDX."""
    t = preprocess(text)
    words = t.split()
    oov = []
    for w in words:
        if len(w) >= 2 and w not in VOCAB_IDX and w not in oov:
            oov.append(w)
    return oov


def add_vocab_words(words):
    """Thêm từ mới hoàn toàn vào VOCAB/VOCAB_IDX, lưu lại dynamic_vocab.json.
    Trả về số từ thực sự thêm được (k)."""
    global VOCAB_SIZE
    added = 0
    for w in words:
        if len(w) >= 2 and w not in VOCAB_IDX:
            VOCAB_IDX[w] = len(VOCAB)
            VOCAB.append(w)
            DYNAMIC_VOCAB.append(w)
            added += 1
    if added:
        VOCAB_SIZE = len(VOCAB)
        with open(DYNAMIC_VOCAB_PATH, "w", encoding="utf-8") as f:
            json.dump(DYNAMIC_VOCAB, f, ensure_ascii=False)
    return added


# Mức "liên quan rõ" theo thang weight mô tả trong lexicon.py (1.0-1.5 yếu,
# 2.0-2.5 rõ, 3.0 mạnh nhất) — dùng làm weight mặc định cho cụm mới học.
PHRASE_DEFAULT_WEIGHT = 2.0


def find_new_phrases(text, oov_words):
    """Tìm cụm 2 từ (bigram) trong câu có chứa ít nhất 1 từ thuộc oov_words
    (từ hoàn toàn mới, do find_oov_words() phát hiện). Đây chính là cụm
    cảm xúc mới mà rule scorer cần học cùng lúc với add_vocab_words(): nếu
    chỉ thêm từ đơn vào VOCAB thì rule_score() (chỉ so khớp theo LEXICON)
    vẫn không biết gì về từ đó, chỉ MLP embedding học được."""
    if not oov_words:
        return []
    t = preprocess(text)
    words = t.split()
    oov_set = set(oov_words)
    phrases = []
    for i in range(len(words) - 1):
        if words[i] in oov_set or words[i+1] in oov_set:
            phrase = words[i] + " " + words[i+1]
            if phrase not in phrases:
                phrases.append(phrase)
    return phrases


def add_lexicon_phrases(phrases, emotion, weight=PHRASE_DEFAULT_WEIGHT):
    """Thêm cụm từ mới vào LEXICON[emotion] (rule scorer học được ngay) và
    vào VOCAB (MLP embedding học được), lưu lại dynamic_lexicon.json.
    Trả về số ô embedding mới cần thêm (để cộng vào k của expand_vocab)."""
    global VOCAB_SIZE
    added = 0
    for phrase in phrases:
        if phrase not in LEXICON[emotion]:
            LEXICON[emotion][phrase] = weight
            DYNAMIC_LEXICON.setdefault(emotion, {})[phrase] = weight
        if phrase not in VOCAB_IDX:
            VOCAB_IDX[phrase] = len(VOCAB)
            VOCAB.append(phrase)
            added += 1
    if added:
        VOCAB_SIZE = len(VOCAB)
    if phrases:
        with open(DYNAMIC_LEXICON_PATH, "w", encoding="utf-8") as f:
            json.dump(DYNAMIC_LEXICON, f, ensure_ascii=False)
    return added


def expand_weights(E, k, d):
    """Dynamic Padding (Ý tưởng 1): np.vstack thêm k hàng mới vào ma trận
    Embedding E, khởi tạo He Initialization x 0.01 để không phá vỡ các đặc
    trưng đã học trước đó."""
    new_rows = np.random.randn(k, d) * np.sqrt(2.0 / d) * 0.01
    return np.vstack([E, new_rows])


# ─── ATTENTION MLP (tự viết: forward + backprop) ─────────────────
class AttentionMLP:
    """
    Kiến trúc:
      Embedding(VOCAB_SIZE, d) -> Self-Attention(Q,K,V) -> mean-pool
        -> Dense(d -> hidden, ReLU) -> Dense(hidden -> N, Softmax)

    Attention(Q,K,V) = Softmax(Q K^T / sqrt(d)) V

    Train: SGD + Momentum (mu=0.9) + L2 (1e-4), chain rule thủ công ngược
    toàn bộ pipeline (CE+Softmax -> Dense -> mean-pool -> attention -> Q/K/V
    -> Embedding rows, gradient embedding accumulate theo từng token_id).
    Lib: chỉ numpy.
    """
    def __init__(self, lr=0.05, d=32, hidden=64):
        self.lr = lr
        self.d = d
        self.hidden = hidden
        self.E  = np.random.randn(VOCAB_SIZE, d) * np.sqrt(2.0 / VOCAB_SIZE)
        self.Wq = np.random.randn(d, d) * np.sqrt(2.0 / d)
        self.Wk = np.random.randn(d, d) * np.sqrt(2.0 / d)
        self.Wv = np.random.randn(d, d) * np.sqrt(2.0 / d)
        self.W1 = np.random.randn(d, hidden) * np.sqrt(2.0 / d)
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, N) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(N)
        # Momentum
        self._params = ["E", "Wq", "Wk", "Wv", "W1", "b1", "W2", "b2"]
        self.m = {p: 0 for p in self._params}
        self.mu = 0.9
        # Adaptive L2 regularization: l2 tăng dần theo feedback_count
        # (xem update_l2) để tránh overfit khi học online nhiều lần
        self.l2_base = 1e-4
        self.l2 = self.l2_base

    def forward(self, token_ids):
        self._ids = token_ids
        if not token_ids:
            self._empty = True
            self._out = np.ones(N) / N
            return self._out
        self._empty = False

        # Tự vá nếu E bị lệch nhỏ hơn VOCAB hiện tại (vd: process restart nạp
        # lại weights.npz cũ hơn dynamic_vocab.json/dynamic_lexicon.json đã
        # ghi xuống đĩa) — tránh IndexError ở backward() khi gặp token mới.
        if self.E.shape[0] < VOCAB_SIZE:
            self.expand_vocab(VOCAB_SIZE - self.E.shape[0])

        X = self.E[token_ids]                        # (T, d)
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv

        scores = (Q @ K.T) / np.sqrt(self.d)          # (T, T)
        scores = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        attn = e / e.sum(axis=-1, keepdims=True)      # (T, T)
        out = attn @ V                                # (T, d)
        c = out.mean(axis=0)                          # (d,) context vector

        z1 = c @ self.W1 + self.b1
        a1 = np.where(z1 > 0, z1, 0.01 * z1)          # Leaky ReLU (slope 0.01)
        z2 = a1 @ self.W2 + self.b2
        ez = np.exp(z2 - z2.max())
        p = ez / ez.sum()                             # Softmax

        self._X, self._Q, self._K, self._V = X, Q, K, V
        self._attn, self._V_out, self._c = attn, out, c
        self._z1, self._a1, self._out = z1, a1, p
        return p

    def backward(self, y_true):
        if self._empty:
            return
        T, d = self._X.shape

        # Dense head: CE + Softmax -> Dense2 -> ReLU -> Dense1
        dz2 = self._out - y_true                                   # (N,)
        dW2 = np.outer(self._a1, dz2) + self.l2 * self.W2
        db2 = dz2
        da1 = self.W2 @ dz2                                         # (hidden,)
        dz1 = da1 * np.where(self._z1 > 0, 1.0, 0.01)               # Leaky ReLU grad
        dW1 = np.outer(self._c, dz1) + self.l2 * self.W1
        db1 = dz1
        dc  = self.W1 @ dz1                                         # (d,)

        # Mean-pool backward: c = mean(out, axis=0) -> đều chia cho T
        d_out = np.tile(dc / T, (T, 1))                             # (T, d)

        # out = attn @ V
        dV    = self._attn.T @ d_out                                # (T, d)
        d_attn = d_out @ self._V.T                                  # (T, T)

        # Softmax backward (row-wise): dscores = attn * (d_attn - sum(d_attn*attn))
        dscores = self._attn * (d_attn - (d_attn * self._attn).sum(axis=-1, keepdims=True))

        # scores = Q @ K.T / sqrt(d)
        dQ = (dscores @ self._K) / np.sqrt(d)                       # (T, d)
        dK = (dscores.T @ self._Q) / np.sqrt(d)                     # (T, d)

        dWq = self._X.T @ dQ + self.l2 * self.Wq
        dWk = self._X.T @ dK + self.l2 * self.Wk
        dWv = self._X.T @ dV + self.l2 * self.Wv
        dX  = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T      # (T, d)

        # Embedding gradient: scatter-accumulate theo token_id
        dE = np.zeros_like(self.E)
        for t, tok in enumerate(self._ids):
            dE[tok] += dX[t]
        dE += self.l2 * self.E

        grads = {"E": dE, "Wq": dWq, "Wk": dWk, "Wv": dWv,
                 "W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

        # Gradient clipping theo global norm: kiến trúc attention thuần
        # không có LayerNorm nên dễ bị nổ gradient khi cộng dồn qua momentum
        total_norm = np.sqrt(sum(np.sum(g*g) for g in grads.values()))
        max_norm = 5.0
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for name in grads:
                grads[name] = grads[name] * scale

        for name, grad in grads.items():
            self.m[name] = self.mu * self.m[name] - self.lr * grad
            setattr(self, name, getattr(self, name) + self.m[name])

    def predict(self, token_ids):
        return self.forward(token_ids)

    def expand_vocab(self, k):
        """Ý tưởng 1: Dynamic Padding - mở rộng Embedding Matrix khi VOCAB
        tăng thêm k từ mới."""
        if k > 0:
            self.E = expand_weights(self.E, k, self.d)
            self.m["E"] = 0  # reset momentum, shape cũ không còn khớp E mới

    def update_l2(self, feedback_count):
        """Adaptive L2 Weight Decay (v2.5): hệ số phạt tăng tỷ lệ thuận với
        số lần feedback (mức độ tương tác online learning của người dùng),
        chặn ở 5x giá trị gốc để tránh "đông cứng" trọng số khi feedback
        còn ít, đồng thời chống overfit khi model được train lại nhiều lần
        trên các mẫu nhỏ."""
        self.l2 = min(self.l2_base * 5, self.l2_base * (1 + feedback_count / 100))

    def save(self, path):
        np.savez(path, E=self.E, Wq=self.Wq, Wk=self.Wk, Wv=self.Wv,
                 W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path):
        if not os.path.exists(path + ".npz"):
            return False
        d = np.load(path + ".npz")
        if "E" not in d.files:
            return False  # định dạng cũ (BoW + MLP), không tương thích
        # Khi đổi kiến trúc (d hoặc hidden thay đổi), bỏ qua weights cũ thay
        # vì crash — engine sẽ pretrain lại từ SEED_DATA + replay buffer.
        if d["E"].shape[1] != self.d or d["W1"].shape[1] != self.hidden:
            print(f"[Load] Kiến trúc thay đổi (saved d={d['E'].shape[1]}/h={d['W1'].shape[1]}, "
                  f"current d={self.d}/h={self.hidden}) — bỏ qua weights cũ, pretrain lại.")
            return False
        self.E, self.Wq, self.Wk, self.Wv = d["E"], d["Wq"], d["Wk"], d["Wv"]
        self.W1, self.b1, self.W2, self.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        self.d = self.E.shape[1]
        return True


# ─── CONTEXTUAL BOOST — sad / lonely / nostalgic disambiguation ──
# Ba class này chia sẻ nhiều từ vựng (trống rỗng, nhớ, một mình) nên dễ
# nhầm lẫn nhau (sad holdout accuracy chỉ 61.3% sau retrain 2026-06-25).
# Giải pháp: khi câu có từ chỉ thời gian quá khứ rõ ràng → boost nostalgic;
# khi có từ chỉ cô lập xã hội → boost lonely; không có tín hiệu nào → giữ
# nguyên để sad/rule scorer tự quyết.
_IDX_NOSTALGIC = EMOTIONS.index("nostalgic")
_IDX_LONELY    = EMOTIONS.index("lonely")

NOSTALGIC_TIME_MARKERS = frozenset({
    "hồi xưa", "ngày xưa", "năm đó", "hồi nhỏ", "hồi bé", "lúc nhỏ",
    "lúc bé", "ngày ấy", "hồi đó", "ngày trước", "khi xưa", "thuở nhỏ",
    "thời đó", "lâu lắm rồi", "mấy năm trước", "hồi còn nhỏ",
})
LONELY_SOCIAL_MARKERS = frozenset({
    "không ai", "một mình", "bơ vơ", "lẻ loi", "giữa đám đông",
    "không có ai", "không ai hiểu", "đơn độc", "cô độc",
    "không ai quan tâm", "chẳng ai", "chẳng có ai",
})


def _contextual_boost(text, scores):
    """Tăng nhẹ điểm nostalgic/lonely khi có tín hiệu ngữ cảnh đặc trưng,
    giúp phân biệt với sad trong các câu có từ vựng chồng lấp."""
    t = preprocess(text)
    modified = scores.copy()
    for m in NOSTALGIC_TIME_MARKERS:
        if m in t:
            modified[_IDX_NOSTALGIC] *= 1.3
            break
    for m in LONELY_SOCIAL_MARKERS:
        if m in t:
            modified[_IDX_LONELY] *= 1.3
            break
    total = modified.sum()
    return modified / total if total > 0 else modified


# ─── HYBRID ENGINE ───────────────────────────────────────────────
class EmotionEngine:
    """
    Kết hợp Rule Scorer + Attention MLP Learner.
    alpha: trọng số cho rule (giảm dần khi có nhiều feedback → tin MLP hơn).
    """
    def __init__(self, weights_path="weights"):
        self.mlp = AttentionMLP(lr=0.02, d=64, hidden=128)
        self.weights_path = weights_path
        self.feedback_count = 0
        self.alpha = 0.85  # ban đầu tin rule nhiều hơn
        # Experience replay: lưu (text, label) các feedback để train lại,
        # tránh "quên" kiến thức cũ khi học câu mới (catastrophic forgetting)
        self.replay = []
        self.replay_path = weights_path + "_replay.json"
        self._load_replay()

        loaded = self.mlp.load(weights_path)
        meta_path = weights_path + "_meta.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                    self.feedback_count = meta.get("feedback_count", 0)
                    self.alpha = self._calc_alpha()
            except Exception:
                pass

        # Adaptive L2 (v2.5): khôi phục hệ số phạt theo feedback_count đã lưu
        self.mlp.update_l2(self.feedback_count)

        if loaded:
            print(f"[Engine] Loaded (attention_v1) | feedback={self.feedback_count} | alpha={self.alpha:.2f} | replay={len(self.replay)}")
        else:
            # Kiến trúc mới (hoặc lần đầu chạy) -> train lại từ SEED_DATA + replay
            print("[Engine] Khởi tạo kiến trúc Self-Attention mới - pretrain từ SEED_DATA + replay buffer...")
            data = SEED_DATA + NEGATION_AUGMENT_DATA + ANGER_AUGMENT_DATA + SAD_AUGMENT_DATA + ROMANTIC_AUGMENT_DATA + [(t, l) for t, l in self.replay]
            self.pretrain(data, epochs=400)

    @property
    def vocab_size(self):
        return len(VOCAB)

    def _calc_alpha(self):
        """Alpha giảm mượt theo số feedback (sigmoid-like), chặn ở [0.35, 0.85].
        Càng nhiều feedback → càng tin MLP hơn rule."""
        fc = self.feedback_count
        return max(0.35, 0.85 - 0.5 * (fc / (fc + 40)))

    def _load_replay(self):
        if os.path.exists(self.replay_path):
            try:
                with open(self.replay_path, encoding="utf-8") as f:
                    self.replay = json.load(f)
            except Exception:
                self.replay = []

    def _save_replay(self):
        # giới hạn 500 mẫu gần nhất để file không phình to
        self.replay = self.replay[-500:]
        with open(self.replay_path, "w", encoding="utf-8") as f:
            json.dump(self.replay, f, ensure_ascii=False)

    def predict(self, text):
        rule = rule_score(text)
        tokens = _tokenize(text)
        ids = [tid for _, tid in tokens if tid is not None]
        mlp  = self.mlp.predict(ids)

        # Hybrid blend + contextual boost (sad/lonely/nostalgic)
        combined = self.alpha * rule + (1 - self.alpha) * mlp
        combined = _contextual_boost(text, combined)
        idx = int(np.argmax(combined))
        emo = EMOTIONS[idx]

        # Knowledge Graph (Ý tưởng 3, nangcap2.txt): attention nhận được của
        # từng token (chuẩn hoá [0,1]) + trọng số cảm xúc theo LEXICON
        graph_tokens = []
        attn = getattr(self.mlp, "_attn", None)
        T = len(ids)
        if attn is not None and T > 0:
            attn_received = attn.sum(axis=0)  # (T,) tổng attention các token khác dồn vào
            max_attn = attn_received.max() or 1.0
            i = 0
            for tok_str, tid in tokens:
                if tid is None:
                    continue
                emos = {e: LEXICON[e][tok_str] for e in EMOTIONS if LEXICON[e].get(tok_str, 0) > 0}
                graph_tokens.append({
                    "text": tok_str,
                    "attention": float(attn_received[i] / max_attn),
                    "emotions": emos,
                })
                i += 1

        return {
            "emotion": emo,
            "label":   EMOTION_META[emo]["vi"],
            "emoji":   EMOTION_META[emo]["emoji"],
            "confidence": float(combined[idx]),
            "all_scores": {EMOTIONS[i]: float(combined[i]) for i in range(N)},
            "rule_scores": {EMOTIONS[i]: float(rule[i]) for i in range(N)},
            "mlp_scores":  {EMOTIONS[i]: float(mlp[i]) for i in range(N)},
            "alpha": self.alpha,
            "graph_tokens": graph_tokens,
        }

    def learn(self, text, correct_emotion, steps=30):
        """Online learning từ feedback người dùng + experience replay.
        Mỗi bước: train câu mới xen kẽ ôn lại vài mẫu cũ → model học câu mới
        nhưng KHÔNG quên kiến thức trước đó. Nếu câu chứa từ hoàn toàn mới
        (OOV), tự động mở rộng VOCAB + Embedding Matrix (Ý tưởng 1)."""
        label = EMOTIONS.index(correct_emotion)

        # Dynamic Weight Expansion: phát hiện + thêm từ mới vào vocab, đồng
        # thời thêm cụm cảm xúc mới (bigram chứa từ mới) vào LEXICON để rule
        # scorer cũng học được — trước đây chỉ add_vocab_words (từ đơn) nên
        # chỉ MLP embedding học, rule scorer không biết gì về cụm từ mới.
        oov = find_oov_words(text)
        added_words = add_vocab_words(oov)
        new_phrases = find_new_phrases(text, oov)
        added_phrases = add_lexicon_phrases(new_phrases, correct_emotion)
        added = added_words + added_phrases
        if added:
            self.mlp.expand_vocab(added)
            print(f"[Vocab] +{added_words} từ mới, +{added_phrases} cụm mới -> vocab_size={self.vocab_size}")

        ids = to_token_ids(text)
        y = np.zeros(N); y[label] = 1.0

        # Lưu vào replay buffer (chống trùng câu y hệt)
        if not any(r[0] == text and r[1] == label for r in self.replay):
            self.replay.append([text, label])

        for step in range(steps):
            # Train câu mới (nhấn mạnh hơn — lặp nhiều lần)
            self.mlp.forward(ids); self.mlp.backward(y)
            # Xen kẽ ôn lại mẫu cũ ngẫu nhiên (replay)
            if self.replay and step % 2 == 0:
                rt, rl = self.replay[np.random.randint(len(self.replay))]
                rids = to_token_ids(rt)
                ry = np.zeros(N); ry[rl] = 1.0
                self.mlp.forward(rids); self.mlp.backward(ry)

        self.feedback_count += 1
        self.alpha = self._calc_alpha()
        self.mlp.update_l2(self.feedback_count)  # Adaptive L2 (v2.5)
        self.save()
        self._save_replay()
        print(f"[Learn] '{text}' -> {correct_emotion} | alpha={self.alpha:.2f} | l2={self.mlp.l2:.6f} | replay={len(self.replay)}")

    def pretrain(self, data, epochs=200):
        """Pre-train Attention MLP trên seed data với learning-rate decay."""
        print(f"[Pretrain] {epochs} epochs x {len(data)} samples (Self-Attention)")
        data = list(data)
        base_lr = self.mlp.lr
        for ep in range(epochs):
            # LR decay: giảm dần để hội tụ ổn định hơn
            self.mlp.lr = base_lr * (1.0 - 0.5 * ep / epochs)
            np.random.shuffle(data)
            loss = 0
            n_valid = 0
            for text, label in data:
                ids = to_token_ids(text)
                if not ids:
                    continue
                y = np.zeros(N); y[label] = 1.0
                p = self.mlp.forward(ids)
                loss += -np.sum(y * np.log(p + 1e-9))
                self.mlp.backward(y)
                n_valid += 1
            if (ep+1) % 50 == 0:
                print(f"  ep {ep+1} loss={loss/max(n_valid,1):.3f}")
        self.mlp.lr = base_lr
        # Đưa toàn bộ data vào replay buffer để feedback sau ôn cùng kiến thức gốc
        self.replay = [[t, l] for t, l in data]
        self.save()
        self._save_replay()

    def save(self):
        self.mlp.save(self.weights_path)
        with open(self.weights_path + "_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "feedback_count": self.feedback_count,
                "alpha": self.alpha,
                "arch": "attention_v1",
                "embed_dim": self.mlp.d,
                "vocab_size": self.vocab_size,
                "l2": self.mlp.l2,
            }, f)


# ─── SEED TRAINING DATA ──────────────────────────────────────────
SEED_DATA = [
    ("hôm nay tôi rất vui", 0), ("hạnh phúc quá trời", 0),
    ("tuyệt vời thật sự", 0), ("vui lắm luôn á", 0),
    ("phấn khởi cả ngày", 0), ("cười suốt không thôi", 0),
    ("thích thú lắm nha", 0), ("mừng vui khôn tả", 0),
    # Trạng thái vui nội tâm — phân biệt rõ với energetic
    ("tâm trạng tốt quá hôm nay", 0), ("mood vui không hiểu sao", 0),
    ("mừng rỡ khi nghe tin tốt", 0), ("phấn chấn lên rồi", 0),
    ("hài lòng với cuộc sống này", 0), ("vui mừng khôn xiết", 0),
    ("toại nguyện với những gì đang có", 0), ("nhẹ nhõm sau khi xong việc", 0),
    ("tươi tắn cả ngày hôm nay", 0), ("rạng ngời khi gặp điều tốt", 0),
    ("niềm vui nho nhỏ thật sự", 0), ("sướng rơn khi nghe tin vui", 0),
    ("thỏa mãn với kết quả đạt được", 0), ("vui bụng cả ngày rồi", 0),

    ("buồn lắm không muốn làm gì", 1), ("khóc cả đêm rồi", 1),
    ("đau lòng vô cùng", 1), ("chán nản thất vọng", 1),
    ("mệt mỏi và tủi thân", 1), ("tan vỡ rồi còn gì nữa", 1),
    ("không muốn sống tiếp", 1), ("tuyệt vọng hoàn toàn", 1),
    # Buồn im lặng / resignation — dạng ít được train nhất
    ("ngậm ngùi nhìn mọi thứ trôi qua", 1), ("chạnh lòng không biết nói sao", 1),
    ("u sầu cả ngày không muốn nói chuyện", 1), ("không thiết gì nữa hết", 1),
    ("buông xuôi rồi không cố nữa", 1), ("vỡ mộng hoàn toàn rồi", 1),
    ("hụt hẫng khi mọi thứ không như kỳ vọng", 1), ("không muốn gặp ai hết", 1),
    ("thổn thức không biết làm sao", 1), ("nghẹn ngào không nói được gì", 1),
    ("đau đáu mãi không nguôi", 1), ("chẳng thiết làm gì cả", 1),
    ("mặc kệ hết rồi không quan tâm nữa", 1), ("cay mắt mà không khóc được", 1),
    # Buồn mất mát / lặng lẽ nội tâm — dạng khó nhận diện nhất
    ("ngồi một mình nhìn mưa lòng nặng trĩu", 1),
    ("thất vọng về chính mình quá nhiều", 1),
    ("không còn hy vọng vào bất cứ điều gì nữa", 1),
    ("đêm nay lòng buồn không hiểu vì sao", 1),
    ("nhớ người đó mà buồn thắt lòng", 1),
    ("mất đi thứ quan trọng không lấy lại được", 1),
    ("chẳng ai biết tôi đang buồn đến thế nào", 1),
    ("tự trách mình mãi vì sai lầm đó", 1),
    ("lòng trống rỗng không biết điền gì vào", 1),
    ("khóc mà không biết lý do tại sao", 1),

    ("đang yêu người đó lắm", 2), ("nhớ người yêu quá", 2),
    ("tim đập nhanh mỗi khi gặp", 2), ("lãng mạn tối nay", 2),
    ("crush nhắn tin rồi", 2), ("yêu thương ngọt ngào", 2),
    ("hẹn hò buổi tối", 2), ("si mê cô ấy quá", 2),

    ("muốn party nhảy múa", 3), ("tập gym thấy cực phê", 3),
    ("năng lượng tràn trề hôm nay", 3), ("nhạc mạnh cho tôi nghe", 3),
    ("sôi nổi hào hứng lắm", 3), ("workout buổi sáng xong", 3),
    ("hype quá muốn nhảy", 3), ("chạy bộ xong thấy đỉnh", 3),

    ("muốn thư giãn nhẹ nhàng", 4), ("ngồi uống trà nghe mưa", 4),
    ("nghỉ ngơi bình yên thôi", 4), ("lofi chill nhẹ nhàng", 4),
    ("yên tĩnh ấm áp cozy", 4), ("relax cuối tuần rồi", 4),
    ("thư thái tâm hồn lắm", 4), ("bình yên không muốn đi đâu", 4),

    ("cô đơn không ai hiểu tôi", 5), ("một mình đêm khuya", 5),
    ("nhớ nhà quá muốn về", 5), ("lạc lõng giữa đám đông", 5),
    ("bơ vơ không ai bên cạnh", 5), ("đêm vắng lặng cô đơn", 5),
    ("xa nhà đã lâu rồi", 5), ("lẻ loi một mình thật sự", 5),

    ("stress deadline quá tải rồi", 6), ("căng thẳng không biết sao", 6),
    ("áp lực công việc nhiều quá", 6), ("lo lắng thi cử sắp tới", 6),
    ("tức giận bực bội cả ngày", 6), ("burn out kiệt sức rồi", 6),
    ("không ổn chút nào", 6), ("hồi hộp lo sợ ghê", 6),

    ("cần tập trung học bài", 7), ("làm việc cần nhạc nền", 7),
    ("study mode on rồi", 7), ("code project quan trọng", 7),
    ("nghiên cứu tài liệu cần yên tĩnh", 7), ("deep work cần focus", 7),
    ("ôn thi cần sự tập trung", 7), ("brainstorm ý tưởng mới", 7),

    ("nhớ về kỷ niệm ngày xưa", 8), ("hoài niệm tuổi thơ êm đềm", 8),
    ("bồi hồi nhớ thanh xuân", 8), ("ngày ấy thật đẹp biết bao", 8),
    ("nhạc retro gợi nhớ quá khứ", 8), ("luyến tiếc thời học trò", 8),
    ("bâng khuâng nhớ ngày tháng cũ", 8), ("hồi ức vintage thật đẹp", 8),

    ("tức giận điên người luôn", 9), ("phẫn nộ vì chuyện vô lý này", 9),
    ("giận dữ không thể chấp nhận được", 9), ("nổi điên với thái độ đó", 9),
    ("bực tức cả ngày vì chuyện đó", 9), ("muốn hét lên vì quá ức chế", 9),
    ("căm phẫn tột độ", 9), ("sôi máu vì bị xúc phạm", 9),
    ("thật không thể tin được họ lại làm vậy", 9), ("không ngờ chuyện này lại xảy ra với tôi", 9),
    ("bực bội vô cùng vì bị đối xử như vậy", 9), ("quá tức vì bị coi thường", 9),
    ("phẫn nộ vì bị đối xử bất công", 9), ("cảm thấy bị phản bội và rất tức giận", 9),
    ("không ngờ lại bị đối xử như vậy", 9), ("thái độ đó khiến tôi nổi giận", 9),
    ("bực bội và ức chế cả ngày", 9), ("giận dữ vì bị xúc phạm và khinh thường", 9),
    ("quá vô lý tôi không chịu được nữa", 9), ("cáu tiết vì sự bất công này", 9),
]


# ─── PHỦ ĐỊNH "VUI" -> SAD (v4.1) ────────────────────────────────
# feedback_log.jsonl thực tế chỉ có ~4 mẫu phủ định trực tiếp "vui" (->
# sad, vd "chẳng vui hơn là bao", "không được vui cho lắm") - quá ít để
# MLP tổng quát hoá PATTERN "phủ định + từ tích cực -> sad" sang biến thể
# câu chữ mới: câu "chẳng vui vẻ gì cả" (chưa từng thấy nguyên văn) vẫn bị
# đoán "happy" vì rule_score() đúng nhưng bị np.maximum(...,0) clip về
# uniform khi phủ định hết điểm (xem rule_score), còn MLP tự nó chưa học
# pattern này. Sinh thêm biến thể bằng template x từ x phủ định x hậu tố
# để MLP thấy nhiều cách diễn đạt khác nhau của CÙNG một pattern, thay vì
# chỉ nhớ vài câu cụ thể.
#
# v4.1.1: bản đầu chỉ có template câu DÀI ("hôm nay tôi không vui...") nên
# sau khi train, 2 mẫu production THẬT vốn đã đúng từ trước ("Không vui",
# "Không được vui cho lắm" - dạng NGẮN trần trụi 2-5 từ) lại bị đoán sai
# thành happy - model học pattern phủ định trong NGỮ CẢNH câu dài nhưng
# không tổng quát hoá ngược lại được dạng ngắn (ít token hơn, attention/
# mean-pool có ít tín hiệu xung quanh để dựa vào). Thêm 2 nhóm sinh BẮT
# BUỘC (không random-sample, để không bị bỏ sót) phủ kín toàn bộ tổ hợp
# phủ định x từ tích cực ở dạng ngắn, trước khi mới sinh thêm câu dài đa
# dạng cho phần còn lại.
_NEG_WORDS = ["không", "chẳng", "chả", "chưa"]
_HAPPY_WORDS = ["vui", "vui vẻ", "hạnh phúc", "vui sướng", "phấn khởi",
                "thích thú", "sướng", "mừng", "vui tươi", "yêu đời"]
_NEG_SUFFIXES = ["", " lắm", " gì cả", " tí nào", " chút nào", " nổi", " mấy", " đâu"]
_NEG_TEMPLATES = [
    "hôm nay tôi {neg} {hw}{suf}",
    "thật ra tôi {neg} {hw}{suf}",
    "chuyện đó làm tôi {neg} {hw}{suf}",
    "dạo này {neg} thấy {hw}{suf}",
    "cả ngày nay {neg} {hw}{suf}",
    "trong lòng {neg} {hw}{suf}",
    "tâm trạng {neg} {hw}{suf}",
    "{neg} còn {hw} như trước nữa",
]
# Dạng NGẮN trần trụi (2-3 từ, giống đúng "Không vui" / "Chẳng vui" thật) -
# sinh BẮT BUỘC đủ mọi tổ hợp neg x happy_word, không qua random sampling.
_NEG_BARE_TEMPLATES = ["{neg} {hw}", "{neg} {hw} lắm"]
# Dạng modal "được...cho lắm" - giống đúng "Không được vui cho lắm" thật.
_NEG_MODAL_TEMPLATES = ["{neg} được {hw} cho lắm", "{neg} được {hw}"]


def _generate_negated_happy_examples(n_diverse=80, seed=7):
    """Sinh câu phủ định từ tích cực -> nhãn sad. Luôn phủ kín TOÀN BỘ tổ
    hợp neg x happy_word ở dạng ngắn (bare + modal) trước, rồi mới sinh
    thêm n_diverse câu dài đa dạng theo template x hậu tố ngẫu nhiên."""
    label = EMOTIONS.index("sad")
    examples = []
    combos = set()

    def add(text):
        if text not in combos:
            combos.add(text)
            examples.append((text, label))

    for neg in _NEG_WORDS:
        for hw in _HAPPY_WORDS:
            for tmpl in _NEG_BARE_TEMPLATES + _NEG_MODAL_TEMPLATES:
                add(tmpl.format(neg=neg, hw=hw))

    rng = random.Random(seed)
    attempts = 0
    target = len(examples) + n_diverse
    while len(examples) < target and attempts < n_diverse * 20:
        attempts += 1
        tmpl = rng.choice(_NEG_TEMPLATES)
        neg = rng.choice(_NEG_WORDS)
        hw = rng.choice(_HAPPY_WORDS)
        suf = rng.choice(_NEG_SUFFIXES)
        add(tmpl.format(neg=neg, hw=hw, suf=suf))
    return examples


NEGATION_AUGMENT_DATA = _generate_negated_happy_examples()


# ─── TỨC GIẬN VÌ BỊ ĐỐI XỬ BẤT CÔNG → ANGRY ─────────────────────
# angry 77.8% trên holdout — bị nhầm với stressed vì cả hai đều negative
# high-arousal. Sự khác biệt chính: angry = directed outward (tức VÌ ai đó/
# điều gì), stressed = inward (áp lực, quá tải). Sinh template có chủ thể
# tức giận cụ thể để MLP học pattern "cause → anger" này.
_ANGER_CAUSES = [
    "bị đối xử bất công", "bị xúc phạm", "bị khinh thường",
    "bị phản bội", "bị ăn hiếp", "bị chà đạp",
    "sự vô lý này", "cách đối xử đó", "thái độ đó",
    "bị đổ lỗi oan", "bị nói xấu sau lưng",
]
_ANGER_EXPR = [
    "tức điên người", "phẫn nộ", "giận dữ", "nổi điên",
    "sôi máu", "bực tức vô cùng", "uất ức", "căm phẫn",
    "không thể chấp nhận", "tức không chịu được",
]
_ANGER_TMPL = [
    "tôi {expr} vì {cause}",
    "{cause} khiến tôi {expr}",
    "thật {expr} vì {cause}",
    "đang {expr} vì {cause}",
    "không chịu được vì {cause}",
    "{cause} thật quá đáng",
    "cảm thấy {expr} khi {cause}",
]


def _generate_anger_examples(seed=13):
    label = EMOTIONS.index("angry")
    examples = []
    combos = set()

    def add(text):
        if text not in combos:
            combos.add(text)
            examples.append((text, label))

    rng = random.Random(seed)
    for cause in _ANGER_CAUSES:
        for expr in _ANGER_EXPR:
            add(rng.choice(_ANGER_TMPL).format(cause=cause, expr=expr))
    for expr in _ANGER_EXPR:
        add(f"{expr} không chịu nổi")
        add(f"đang {expr} lắm")
    return examples


ANGER_AUGMENT_DATA = _generate_anger_examples()


# ─── BUỒN VÌ MẤT MÁT / THẤT BẠI → SAD ──────────────────────────
# sad 61.3% holdout — bị nhầm sang angry (sau khi thêm ANGER_AUGMENT) và
# lonely (cùng "một mình"). Sự khác biệt: sad = buồn hướng vào nội tâm
# (mất mát, thất bại, chia tay) không có tức giận và không cô đơn xã hội.
# Sinh template nguyên nhân buồn × cảm xúc buồn để MLP học pattern rõ ràng.
_SAD_CAUSES = [
    "chia tay rồi", "mất đi người thân", "bị từ chối",
    "thất bại rồi", "giấc mơ tan vỡ", "mọi thứ sụp đổ",
    "kết quả không như kỳ vọng", "nỗ lực bao lâu mà vô ích",
    "mất đi thứ quan trọng", "người đó rời đi rồi",
    "cố gắng mà không được công nhận", "bị phụ lòng",
]
_SAD_EXPR = [
    "lòng đau nhói", "buồn thắt lòng", "đau lòng lắm",
    "nước mắt không ngừng", "lòng trống rỗng", "buồn không tả được",
    "tâm trạng rất tệ", "không thiết làm gì", "chán chường vô cùng",
    "sụp đổ hoàn toàn", "kiệt sức cả tâm hồn", "nghẹn ngào cả ngày",
]
_SAD_TMPL = [
    "{cause}, {expr}",
    "{cause} nên {expr}",
    "cảm thấy {expr} vì {cause}",
    "{expr} khi nghĩ đến {cause}",
    "sau khi {cause} tôi {expr}",
    "{expr} vì {cause} rồi",
    "tôi {expr} khi {cause}",
]


def _generate_sad_examples(seed=17):
    label = EMOTIONS.index("sad")
    examples = []
    combos = set()

    def add(text):
        if text not in combos:
            combos.add(text)
            examples.append((text, label))

    rng = random.Random(seed)
    for cause in _SAD_CAUSES:
        for expr in _SAD_EXPR:
            add(rng.choice(_SAD_TMPL).format(cause=cause, expr=expr))
    for expr in _SAD_EXPR:
        add(f"{expr} quá")
        add(f"đang {expr}")
    return examples


SAD_AUGMENT_DATA = _generate_sad_examples()


# ─── LÃNG MẠN VỀ MỘT NGƯỜI CỤ THỂ → ROMANTIC ───────────────────
# romantic 50.9% holdout — tệ nhất, không có augmentation data, bị nhầm với
# happy (vui chung chung), relaxed (êm dịu), nostalgic (nhớ nhung quá khứ).
# Khác biệt cốt lõi: romantic PHẢI có một người cụ thể (người yêu, crush)
# đang hiện diện hoặc được nhắc trực tiếp — không phải cảm giác vui/bình yên
# chung mà không có đối tượng tình cảm.
_ROMANTIC_PERSONS = [
    "người yêu", "anh ấy", "cô ấy", "em ấy", "crush",
    "người đặc biệt đó", "người mình thương", "người ấy",
    "anh", "em",
]
_ROMANTIC_EXPR = [
    "xao xuyến lắm", "tim đập nhanh", "nhớ nhung da diết",
    "yêu lắm lắm", "thương vô cùng", "hồi hộp khi gặp",
    "muốn ở bên mãi", "cảm giác ấm áp lạ lắm",
    "bất giác mỉm cười khi nghĩ đến", "lòng dịu lại khi có",
    "ngại ngùng mà hạnh phúc", "nhớ từng nụ cười của",
]
_ROMANTIC_TMPL = [
    "nghĩ đến {person} thấy {expr}",
    "khi ở bên {person} {expr}",
    "{expr} mỗi khi gặp {person}",
    "chỉ cần có {person} bên cạnh là {expr}",
    "hôm nay gặp {person} thấy {expr}",
    "{person} nhắn tin mà {expr}",
    "nhìn {person} cười thấy {expr}",
    "cùng {person} đi dạo, {expr}",
]


def _generate_romantic_examples(seed=23):
    label = EMOTIONS.index("romantic")
    examples = []
    combos = set()

    def add(text):
        if text not in combos:
            combos.add(text)
            examples.append((text, label))

    rng = random.Random(seed)
    for person in _ROMANTIC_PERSONS:
        for expr in _ROMANTIC_EXPR:
            add(rng.choice(_ROMANTIC_TMPL).format(person=person, expr=expr))
    for expr in _ROMANTIC_EXPR:
        add(f"đang {expr} ghê")
        add(f"sao mà {expr} thế này")
    return examples


ROMANTIC_AUGMENT_DATA = _generate_romantic_examples()


# ─── MAIN ────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    engine = EmotionEngine("weights")

    tests = [
        "hôm nay tôi rất vui và hạnh phúc",
        "buồn quá không muốn làm gì cả",
        "yêu em lắm nhớ em nhiều",
        "muốn tập gym nghe nhạc sôi động",
        "ngồi uống cà phê nghe lofi chill",
        "cô đơn một mình trong đêm khuya",
        "stress deadline quá tải không chịu được",
        "cần tập trung học bài thi quan trọng",
        "nhớ về kỷ niệm tuổi thơ ngày xưa",
        "tức giận điên người vì bị phản bội",
    ]
    print("\n[Test Results]")
    print("-"*65)
    for t in tests:
        r = engine.predict(t)
        bar = "█" * int(r["confidence"]*20)
        print(f"Input : {t}")
        print(f"Result: {r['emoji']} {r['label']} | {r['confidence']*100:.1f}% {bar}")
        print()
