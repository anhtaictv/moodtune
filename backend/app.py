from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json, os, sys, time, random, threading
from datetime import datetime
from collections import Counter
import urllib.request, urllib.parse, urllib.error

# Ép stdout/stderr dùng UTF-8 để print tiếng Việt không crash trên Windows (cp1252)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(__file__))
from emotion_mlp import EmotionEngine, EMOTIONS, EMOTION_META
from bandit import ThompsonBandit
import audio_features

app = Flask(__name__)
_secret_key = os.environ.get("MOODTUNE_SECRET_KEY") or ""
if not _secret_key:
    import warnings
    warnings.warn("[SECURITY] MOODTUNE_SECRET_KEY not set — using insecure default. Set the env var in production.")
    _secret_key = "moodtune_secret_2024"
app.secret_key = _secret_key

# ─── CORS: Cho phép origin của frontend, KHÔNG kèm credentials ────
# Trước đây code dùng supports_credentials=True + origin "*" → trình duyệt
# từ chối combo này theo CORS spec. Bây giờ liệt kê origin rõ ràng
# và bỏ credentials (frontend không cần cookies).
FRONTEND_URL = os.environ.get("MOODTUNE_FRONTEND", "https://anhtaictv.me")
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost",
    "http://localhost:5500",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:8080",
]
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=False)

# ─── JAMENDO API (FREE - FULL TRACKS, NO USER LOGIN) ─────────────
JAMENDO_CLIENT_ID  = os.environ.get("JAMENDO_CLIENT_ID", "cf31dbfd")
JAMENDO_API        = "https://api.jamendo.com/v3.0"
JAMENDO_TIMEOUT    = 12   # giây

# Map genre chip ở UI → tag Jamendo. None nghĩa là không lọc.
GENRE_TAG = {
    "all":      None,
    "vpop":     "pop",          # Jamendo không có vpop riêng → fallback "pop"
    "ballad":   "ballad",
    "lofi":     "lofi",
    "edm":      "electronic",
    "acoustic": "acoustic",
}

JAMENDO_MAX_RETRIES = 3

def jamendo_search(query=None, tags=None, artist=None, limit=30,
                   offset=0, order="popularity_total"):
    """Search Jamendo API - free full tracks, only needs a client_id."""
    p = {
        "client_id":   JAMENDO_CLIENT_ID,
        "format":      "json",
        "limit":       limit,
        "offset":      offset,
        "order":       order,
        "audioformat": "mp32",
        "include":     "musicinfo",
        "imagesize":   "300",
    }
    if query:  p["namesearch"]   = query
    if tags:   p["tags"]         = tags          # ví dụ: "happy+pop"
    if artist: p["artist_name"]  = artist

    params = urllib.parse.urlencode(p)
    req = urllib.request.Request(f"{JAMENDO_API}/tracks?{params}")

    last_err = None
    for attempt in range(JAMENDO_MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=JAMENDO_TIMEOUT) as resp:
                data = json.loads(resp.read())
            break
        except (urllib.error.URLError, OSError) as e:
            last_err = e
            if attempt < JAMENDO_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    else:
        raise last_err

    tracks = []
    for item in data.get("results", []):
        # Gom tags nhạc (genre/mood) để phục vụ gợi ý thông minh
        mi = item.get("musicinfo", {}) or {}
        tag_list = []
        if isinstance(mi.get("tags"), dict):
            for v in mi["tags"].values():
                if isinstance(v, list):
                    tag_list += v
        tracks.append({
            "id":        str(item["id"]),
            "name":      item["name"],
            "artist":    item.get("artist_name", ""),
            "album":     item.get("album_name", ""),
            "image":     item.get("album_image") or item.get("image", ""),
            "preview":   item.get("audio"),          # full track MP3 stream
            "duration":  item.get("duration", 0),
            "open_url":  item.get("shareurl", ""),
            "tags":      tag_list[:6],
        })
    return tracks

# Mỗi cảm xúc có nhiều cặp tag → mỗi lần random để kết quả luôn mới & đa dạng
# Tag Jamendo ánh xạ theo mô hình Valence-Arousal (GEMS/Circumplex,
# xem thongtin.docx + EMOTION_META.valence/arousal trong lexicon.py).
EMOTION_TAGS = {
    "happy":     ["happy+bright", "happy+acoustic", "fun+upbeat", "feelgood+happy",
                   "pop+happy", "disco+funk", "reggae+happy"],
    "sad":       ["sad+slow", "melancholic+soul", "sad+melancholic", "sad+acoustic"],
    "romantic":  ["romantic+love", "love+ballad", "romantic+slow", "love+song", "soul+rnb"],
    "energetic": ["energetic+electronic", "energetic+dance", "dance+pop", "electro+upbeat", "power+electro"],
    "relaxed":   ["relax+chillout", "lofi+ambient", "chillout+lounge", "ambient+relax", "downtempo+chill"],
    "lonely":    ["lonely+atmospheric", "ambient+lonely", "melancholic+acoustic", "sad+lonely", "ambient+emotional"],
    "stressed":  ["stress+heavy", "intense+industrial", "rock+intense", "metal+power", "punk+fast"],
    "focused":   ["focus+instrumental", "piano+classical", "ambient+piano", "lofi+chill", "instrumental+calm"],
    "nostalgic": ["nostalgic+vintage", "acoustic+retro", "retro+synthwave", "retro+electronic", "80s+retro"],
    "angry":     ["anger+aggressive", "metal+hardcore", "metal+aggressive", "punk+rock", "rock+heavy"],
}

def pick_emotion_tags(emotion):
    """Chọn random 1 cặp tag từ pool của cảm xúc → kết quả đa dạng."""
    pool = EMOTION_TAGS.get(emotion, ["chillout"])
    return random.choice(pool)

def pick_emotion_tags_bandit(emotion):
    """RLUF (Ý tưởng 1, nangcap2.txt): Thompson Sampling chọn cặp tag thay
    cho random.choice - dần học gu nhạc của người dùng theo từng cảm xúc.
    Trả về (tags, tag_arm_index)."""
    pool = EMOTION_TAGS.get(emotion, ["chillout"])
    idx = mab.sample_tag_index(emotion, len(pool))
    return pool[idx], idx

def merge_genre(base_tags, genre):
    """Nối tag genre người dùng chọn vào tag cơ sở (nếu có)."""
    extra = GENRE_TAG.get((genre or "all").lower())
    if not extra or not base_tags:
        return base_tags
    # Tránh nối trùng nếu đã có
    if extra in base_tags.split("+"):
        return base_tags
    return f"{base_tags}+{extra}"

def intensity_to_order(intensity):
    """Slider Cường độ 1-10 → thay đổi tiêu chí xếp hạng kết quả Jamendo.
    Thấp = nhạc mới nổi tháng này (mellow), giữa = all-time, cao = trending tuần."""
    try:
        v = int(intensity)
    except (TypeError, ValueError):
        return "popularity_total"
    if v <= 3:  return "popularity_month"
    if v >= 8:  return "popularity_week"
    return "popularity_total"

def time_to_emotion(hour):
    """Map giờ trong ngày → cảm xúc gợi ý hợp lý."""
    if 5 <= hour < 11:   return "energetic"
    if 11 <= hour < 14:  return "focused"
    if 14 <= hour < 18:  return "happy"
    if 18 <= hour < 22:  return "relaxed"
    return "relaxed"

# ─── AI ENGINE ────────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights")
engine = EmotionEngine(WEIGHTS_PATH)
# AttentionMLP stores forward-pass state as instance attributes (self._ids,
# self._attn, …) and mutates E/W1/W2 in-place during backward(). Two concurrent
# requests touching these would corrupt each other's state, so all engine
# calls are serialised behind this lock before bumping waitress to threads>1.
_engine_lock = threading.Lock()

# ─── RLUF BANDIT (Thompson Sampling - v3.0) ───────────────────────
BANDIT_PATH = os.path.join(os.path.dirname(__file__), "bandit_state.json")
mab = ThompsonBandit(BANDIT_PATH)

LOG_PATH = os.path.join(os.path.dirname(__file__), "feedback_log.jsonl")
LOG_MAX_SIZE_BYTES = 20 * 1024 * 1024  # rotate at 20 MB
LOG_ROTATE_CHECK   = 100               # check size every N writes
_log_write_count   = 0

# ─── RATE LIMIT cho /api/learn (chống spam đầu độc model online learning) ──
# Online learning dùng 1 model NumPy chung cho mọi người dùng (không phải
# user-riêng) -> phải chặn 1 IP spam nhãn sai để bảo vệ model chung, nhưng
# vẫn cho script dạy AI nội bộ (gemini_teacher.py) bỏ qua giới hạn qua
# header bí mật MOODTUNE_ADMIN_KEY (không set -> bypass này tắt hẳn).
MOODTUNE_ADMIN_KEY = os.environ.get("MOODTUNE_ADMIN_KEY", "")
LEARN_RATE_LIMIT  = 12   # tối đa số request /api/learn
LEARN_RATE_WINDOW = 60   # ... trong mỗi cửa sổ (giây) / 1 IP

_learn_hits_lock = threading.Lock()
_learn_hits = {}   # ip -> [timestamp request gần đây]

def _client_ip():
    fwd = request.headers.get("X-Forwarded-For", "")
    return fwd.split(",")[0].strip() if fwd else (request.remote_addr or "unknown")

def _is_admin_request():
    return bool(MOODTUNE_ADMIN_KEY) and request.headers.get("X-Admin-Key") == MOODTUNE_ADMIN_KEY

def _is_rate_limited(ip):
    now = time.time()
    with _learn_hits_lock:
        hits = [t for t in _learn_hits.get(ip, []) if now - t < LEARN_RATE_WINDOW]
        hits.append(now)
        _learn_hits[ip] = hits
        return len(hits) > LEARN_RATE_LIMIT

def _maybe_rotate_log():
    try:
        if os.path.getsize(LOG_PATH) > LOG_MAX_SIZE_BYTES:
            os.replace(LOG_PATH, LOG_PATH + ".1")
    except OSError:
        pass

def log_event(event_type, data):
    global _log_write_count
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(
            {"time": datetime.now().isoformat(), "type": event_type, **data},
            ensure_ascii=False
        ) + "\n")
    _log_write_count += 1
    if _log_write_count % LOG_ROTATE_CHECK == 0:
        _maybe_rotate_log()

# ─── BỘ ĐẾM NGƯỜI DÙNG (online / đang nghe / tổng lượt truy cập) ──
# Không dùng DB/WebSocket: frontend gọi /api/presence/ping định kỳ
# (heartbeat), server giữ session trong RAM và coi là "rời trang" nếu
# quá PRESENCE_TIMEOUT giây không thấy ping nào.
VISIT_STATS_PATH  = os.path.join(os.path.dirname(__file__), "visit_stats.json")
PRESENCE_TIMEOUT  = 40   # giây không ping → coi như đã rời trang

presence_lock     = threading.Lock()
online_sessions   = {}   # session_id -> {"last_seen": ts, "listening": bool}

# Baseline "giả lập" sống ở module riêng (backend/fake_presence.py, không
# track trên git — xem .gitignore). Thiếu file này thì coi như baseline = 0,
# bộ đếm tự fallback về số thật 100%, không lỗi.
try:
    import fake_presence
except ImportError:
    fake_presence = None

def _load_visit_total():
    if os.path.exists(VISIT_STATS_PATH):
        try:
            with open(VISIT_STATS_PATH, encoding="utf-8") as f:
                return int(json.load(f).get("total_visits", 0))
        except Exception:
            pass
    return 0

def _save_visit_total(total):
    tmp = VISIT_STATS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"total_visits": total}, f)
    os.replace(tmp, VISIT_STATS_PATH)

visit_total = _load_visit_total()

# Tổng lượt truy cập tăng theo mức tăng của số online hiển thị (thật + giả
# lập gộp lại) — online tăng bao nhiêu thì cộng bấy nhiêu vào tổng, online
# giảm thì không trừ và không cộng. Nhờ vậy mọi lượt tăng online (do người
# thật vào hoặc do baseline dao động) đều phản ánh vào tổng lượt truy cập.
last_online_count = None
_last_prune_time  = 0.0
PRUNE_INTERVAL    = 10  # seconds between expensive session scans

def _prune_sessions():
    global _last_prune_time
    now = time.time()
    if now - _last_prune_time < PRUNE_INTERVAL:
        return
    _last_prune_time = now
    cutoff = now - PRESENCE_TIMEOUT
    for sid in [s for s, v in online_sessions.items() if v["last_seen"] < cutoff]:
        online_sessions.pop(sid, None)

# ─── ENDPOINTS ────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    resp = jsonify({
        "status": "ok",
        "model": "EmotionMLP-Hybrid",
        "music_api": "jamendo_free",
        "frontend": FRONTEND_URL,
        "feedback_count": engine.feedback_count,
        "alpha": engine.alpha,
        "emotions": EMOTIONS,
        # ⬇ Thêm thông tin kiến trúc thật để frontend đồng bộ
        "architecture": {
            "vocab_size":  engine.vocab_size,
            "embed_dim":   engine.mlp.d,
            "hidden_size": engine.mlp.W1.shape[1],
            "output_size": len(EMOTIONS),
            "attention":   True,
            "activation":  "leaky_relu",
            "l2":          round(engine.mlp.l2, 6),
        },
    })
    resp.headers["Cache-Control"] = "public, max-age=30"
    return resp

@app.route("/api/presence/ping", methods=["POST"])
def presence_ping():
    """Heartbeat từ frontend (gọi định kỳ + khi rời trang qua sendBeacon).
    Trả về số người đang online / đang nghe nhạc / tổng lượt truy cập."""
    global visit_total, last_online_count
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    fake_online, fake_listening = fake_presence.baseline() if fake_presence else (0, 0)
    with presence_lock:
        if data.get("leaving"):
            online_sessions.pop(session_id, None)
        else:
            _prune_sessions()
            online_sessions[session_id] = {
                "last_seen": time.time(),
                "listening": bool(data.get("listening")),
            }
        real_online    = len(online_sessions)
        real_listening = sum(1 for v in online_sessions.values() if v["listening"])
        online_count    = fake_online + real_online
        listening_count = fake_listening + real_listening
        if last_online_count is None:
            last_online_count = online_count
        elif online_count > last_online_count:
            visit_total += online_count - last_online_count
            _save_visit_total(visit_total)
        last_online_count = online_count

    return jsonify({
        "status": "ok", "online": online_count,
        "listening": listening_count, "total_visits": visit_total,
    })

TEXT_MAX_LEN = 1000

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "text required"}), 400
    if len(text) > TEXT_MAX_LEN:
        return jsonify({"error": f"text too long (max {TEXT_MAX_LEN} chars)"}), 400
    with _engine_lock:
        result = engine.predict(text)
    scores_list = sorted(
        [{"emotion": e, "label": EMOTION_META[e]["vi"],
          "emoji": EMOTION_META[e]["emoji"], "score": result["all_scores"][e],
          "valence": EMOTION_META[e]["valence"], "arousal": EMOTION_META[e]["arousal"]}
         for e in EMOTIONS], key=lambda x: -x["score"]
    )
    log_event("predict", {"text": text, "result": result["emotion"]})
    emo = result["emotion"]
    return jsonify({
        "emotion": emo, "label": result["label"],
        "emoji": result["emoji"], "confidence": result["confidence"],
        "valence": EMOTION_META[emo]["valence"],
        "arousal": EMOTION_META[emo]["arousal"],
        "scores": scores_list,
        "graph": result["graph_tokens"],
        "model_info": {
            "alpha_rule": result["alpha"],
            "alpha_mlp": round(1 - result["alpha"], 2),
            "feedback_count": engine.feedback_count,
        },
    })

@app.route("/api/learn", methods=["POST"])
def learn():
    if not _is_admin_request() and _is_rate_limited(_client_ip()):
        return jsonify({"error": f"Quá nhiều yêu cầu dạy model, thử lại sau "
                                  f"({LEARN_RATE_LIMIT} request/{LEARN_RATE_WINDOW}s)."}), 429
    data    = request.get_json()
    text    = (data or {}).get("text", "").strip()
    correct = (data or {}).get("correct_emotion", "").strip()
    if not text or not correct:
        return jsonify({"error": "text and correct_emotion required"}), 400
    if len(text) > TEXT_MAX_LEN:
        return jsonify({"error": f"text too long (max {TEXT_MAX_LEN} chars)"}), 400
    if correct not in EMOTIONS:
        return jsonify({"error": f"Unknown emotion: {correct}"}), 400
    with _engine_lock:
        engine.learn(text, correct, steps=30)
    log_event("feedback", {"text": text, "correct": correct,
                           "feedback_count": engine.feedback_count})
    return jsonify({
        "status": "learned", "feedback_count": engine.feedback_count,
        "new_alpha": engine.alpha,
        "message": f"Model đã học: '{text}' → {EMOTION_META[correct]['vi']}",
    })

@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """Phân tích cảm xúc theo lô — dùng cho index Kho nhạc Local (Ý tưởng 4).
    Nhận {"items": ["ten_bai_1", ...]} (tối đa 200), trả emotion cho từng item."""
    data  = request.get_json() or {}
    items = data.get("items", [])
    if not isinstance(items, list) or not items:
        return jsonify({"error": "items required"}), 400
    items = items[:200]

    results = []
    for name in items:
        r = engine.predict(str(name).strip())
        results.append({
            "name": name, "emotion": r["emotion"],
            "label": r["label"], "emoji": r["emoji"],
        })
    log_event("predict_batch", {"count": len(results)})
    return jsonify({"results": results})

_stats_cache     = {"data": None, "time": 0.0}
STATS_CACHE_TTL  = 60  # seconds

@app.route("/api/stats")
def stats():
    now = time.time()
    if _stats_cache["data"] and now - _stats_cache["time"] < STATS_CACHE_TTL:
        resp = jsonify(_stats_cache["data"])
        resp.headers["Cache-Control"] = f"public, max-age={STATS_CACHE_TTL}"
        return resp

    s = {"total_predicts": 0, "total_feedback": 0,
         "emotion_counts": {e: 0 for e in EMOTIONS}}
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                    if ev["type"] == "predict":
                        s["total_predicts"] += 1
                        emo_key = ev.get("result", "happy")
                        # Bảo vệ: nếu cảm xúc cũ không còn trong EMOTIONS thì bỏ qua
                        if emo_key in s["emotion_counts"]:
                            s["emotion_counts"][emo_key] += 1
                    elif ev["type"] == "feedback":
                        s["total_feedback"] += 1
                except Exception:
                    pass
    result = {**s, "model_alpha": engine.alpha,
              "feedback_count": engine.feedback_count}
    _stats_cache["data"] = result
    _stats_cache["time"] = now
    resp = jsonify(result)
    resp.headers["Cache-Control"] = f"public, max-age={STATS_CACHE_TTL}"
    return resp

# ─── JAMENDO MUSIC SEARCH (NO USER LOGIN) ────────────────────────

@app.route("/api/music/search")
def music_search():
    """Search music trên Jamendo - không cần login.
    Hỗ trợ: emotion, q (từ khoá), offset (phân trang), genre, intensity."""
    emotion   = request.args.get("emotion", "relaxed")
    query     = request.args.get("q", "").strip()
    offset    = int(request.args.get("offset", 0) or 0)
    limit     = int(request.args.get("limit", 30) or 30)
    genre     = request.args.get("genre", "all")
    intensity = request.args.get("intensity", 5)
    order     = request.args.get("order") or intensity_to_order(intensity)

    tag_arm = None
    try:
        if query:
            # Tìm thủ công theo tên bài/nghệ sĩ — không trộn genre/tag để
            # tôn trọng đúng từ khoá user gõ.
            tracks = jamendo_search(query=query, limit=limit,
                                    offset=offset, order=order)
            used = query
        else:
            base_tags, tag_arm = pick_emotion_tags_bandit(emotion)
            tags = merge_genre(base_tags, genre)
            tracks = jamendo_search(tags=tags, limit=limit,
                                    offset=offset, order=order)
            used = tags

            # Ý tưởng 2 (lite): phân tích audio nền + soft re-rank theo cache
            if audio_features.AUDIO_ENABLED:
                audio_features.analyze_async(tracks)
                for t in tracks:
                    cached = audio_features.get_cached(t["id"])
                    if cached:
                        t["audio_emotion"] = cached["audio_emotion"]
                tracks.sort(key=lambda t: 0 if t.get("audio_emotion") == emotion else 1)
        return jsonify({
            "tracks": tracks, "query": query or emotion,
            "tags_used": used, "source": "jamendo",
            "offset": offset, "limit": limit,
            "order": order, "genre": genre,
            "has_more": len(tracks) >= limit,
            "tag_arm": tag_arm,
        })
    except urllib.error.URLError as e:
        print(f"[Jamendo] Network error: {e}")
        return jsonify({"error": "jamendo_unreachable", "tracks": []}), 502
    except Exception as e:
        print(f"[Jamendo] Search error: {e}")
        return jsonify({"error": "internal_error", "tracks": []}), 500

# ─── RLUF: TỈ LỆ TRỘN NHẠC ONLINE/LOCAL (Ý tưởng 1, nangcap2.txt) ─

@app.route("/api/mix-ratio")
def mix_ratio():
    """Thompson Sampling: gợi ý tỉ lệ trộn nhạc Online:Local cho 12 bài
    tiếp theo, dựa trên lịch sử Like/Dislike/Next của người dùng theo
    từng cảm xúc."""
    emotion = request.args.get("emotion", "relaxed")
    n_online, n_local = mab.sample_mix(emotion, total=12)
    return jsonify({
        "emotion": emotion,
        "online": n_online, "local": n_local,
        "ratio": mab.get_summary(emotion),
    })

# ─── LƯU HÀNH VI NGƯỜI DÙNG (GỢI Ý THÔNG MINH) ───────────────────

@app.route("/api/track/event", methods=["POST"])
def track_event():
    """Lưu sự kiện nghe/like/dislike/next để thống kê & gợi ý thông minh.
    Like/Dislike/Next còn được dùng làm reward cho RLUF Bandit (Ý tưởng 1,
    nangcap2.txt) để tối ưu tỉ lệ trộn nhạc Online/Local và chọn tag."""
    d = request.get_json() or {}
    etype = d.get("type", "")
    if etype not in ("play", "like", "dislike", "next"):
        return jsonify({"error": "invalid type"}), 400
    emotion = d.get("emotion", "")
    if emotion and emotion not in EMOTIONS:
        return jsonify({"error": "invalid emotion"}), 400
    source  = d.get("source", "")
    if source and source not in ("online", "local"):
        return jsonify({"error": "invalid source"}), 400
    tag_arm = d.get("tag_arm")
    log_event("track_" + etype, {
        "track_id": d.get("track_id", ""),
        "name":     d.get("name", ""),
        "artist":   d.get("artist", ""),
        "tags":     d.get("tags", []),
        "emotion":  emotion,
        "source":   source,
        "tag_arm":  tag_arm,
    })
    if etype in ("like", "dislike", "next") and emotion:
        reward = 1 if etype == "like" else -1
        mab.update_source(emotion, source or "online", reward)
        if source == "online" and isinstance(tag_arm, int):
            mab.update_tag(emotion, tag_arm, reward)
    return jsonify({"status": "logged"})

def _top_from_log(field, etypes, top_n=5):
    """Đếm giá trị phổ biến nhất của 1 field trong log theo loại sự kiện."""
    counter = Counter()
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") not in etypes:
                    continue
                val = ev.get(field)
                if isinstance(val, list):
                    counter.update([v for v in val if v])
                elif val:
                    counter[val] += 1
    return [k for k, _ in counter.most_common(top_n)]

@app.route("/api/recommend")
def recommend():
    """Gợi ý nhạc dựa trên nghệ sĩ/tag user thích (ưu tiên client) +
    fallback theo giờ. liked_artists/liked_tags: chuỗi phân tách bởi ','."""
    cli_artists = [a for a in request.args.get("liked_artists", "").split(",") if a.strip()]
    cli_tags    = [t for t in request.args.get("liked_tags", "").split(",") if t.strip()]

    # Kết hợp dữ liệu client (ưu tiên) với thống kê chung của server
    artists = cli_artists or _top_from_log("artist", ("track_like",), 3)
    tags    = cli_tags    or _top_from_log("tags",   ("track_like", "track_play"), 3)

    tracks, seen = [], set()
    try:
        # 1) Nhạc của các nghệ sĩ user hay thích
        for a in artists[:3]:
            for t in jamendo_search(artist=a.strip(), limit=6):
                if t["id"] not in seen:
                    seen.add(t["id"]); tracks.append(t)
        # 2) Nhạc theo tag ưa thích
        if tags:
            for t in jamendo_search(tags="+".join(t.strip() for t in tags[:2]), limit=12):
                if t["id"] not in seen:
                    seen.add(t["id"]); tracks.append(t)
        # 3) Fallback: chưa đủ dữ liệu → gợi ý theo giờ
        if len(tracks) < 8:
            emo = time_to_emotion(datetime.now().hour)
            for t in jamendo_search(tags=pick_emotion_tags(emo), limit=15):
                if t["id"] not in seen:
                    seen.add(t["id"]); tracks.append(t)

        random.shuffle(tracks)
        return jsonify({
            "tracks": tracks[:20], "source": "jamendo",
            "based_on": {"artists": artists, "tags": tags},
            "personalized": bool(artists or tags),
        })
    except urllib.error.URLError as e:
        print(f"[Recommend] Network error: {e}")
        return jsonify({"error": "jamendo_unreachable", "tracks": []}), 502
    except Exception as e:
        print(f"[Recommend] error: {e}")
        return jsonify({"error": "internal_error", "tracks": []}), 500

@app.route("/api/time-suggestion")
def time_suggestion():
    """Trả cảm xúc gợi ý theo giờ. Nhận ?hour= từ client (giờ máy người dùng)."""
    h = request.args.get("hour")
    hour = int(h) if (h is not None and h.isdigit()) else datetime.now().hour
    emo = time_to_emotion(hour)
    return jsonify({
        "hour": hour, "emotion": emo,
        "label": EMOTION_META[emo]["vi"],
        "emoji": EMOTION_META[emo]["emoji"],
    })

# ─── RUN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print(" MoodTune AI Backend + Jamendo Music API")
    print(f" Feedback: {engine.feedback_count} | Alpha: {engine.alpha:.2f}")
    print(f" Frontend: {FRONTEND_URL}")
    print(f" Vocab: {engine.vocab_size} | Embed: {engine.mlp.d} | Hidden: {engine.mlp.W1.shape[1]} (Self-Attention)")
    print(f" Music API: Jamendo (Free - Full Tracks)")
    print("=" * 50)
    # waitress thay cho Flask dev server. threads=4: engine calls are now
    # serialised behind _engine_lock so concurrent requests are safe.
    from waitress import serve
    serve(app, host="0.0.0.0", port=5005, threads=4)