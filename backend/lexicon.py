# ═══════════════════════════════════════════════════════════════════
#  lexicon.py — Từ điển cảm xúc MoodTune
#  Chỉnh sửa file này để thêm/bớt từ, KHÔNG cần đụng vào emotion_mlp.py
#
#  Cấu trúc:
#    EMOTIONS      — thứ tự class (giữ nguyên khi đã train, đổi thứ tự sẽ
#                    làm weights.npz cũ không còn đúng class nữa)
#    EMOTION_META  — tên hiển thị + emoji cho từng class
#    LEXICON       — {emotion: {từ/cụm_từ: weight}}
#                    weight 1.0–1.5 : liên quan yếu
#                    weight 2.0–2.5 : liên quan rõ
#                    weight 3.0     : từ đặc trưng mạnh nhất
#    NEGATIONS     — từ/cụm phủ định (1-3 từ) đứng ngay trước từ/cụm cảm
#                    xúc (unigram lẫn bigram) → nhân ×-0.6. Hỗ trợ cả phủ
#                    định nhiều từ ("không hề", "chẳng bao giờ") nhờ
#                    rule_score() quét các cụm con 1..N từ ngay trước vị
#                    trí đang xét (xem _is_negated() trong emotion_mlp.py)
#
#  Ghi chú bigram:
#    Cụm 2 từ ("ngày xưa", "tập trung") tự động được boost ×1.5 trong
#    rule_score() — không cần chỉnh code, cứ thêm cụm vào đây là được.
# ═══════════════════════════════════════════════════════════════════

# ── THỨ TỰ CLASS ─────────────────────────────────────────────────
# !! CẢNH BÁO: Không đổi thứ tự sau khi đã train. Thêm class mới thì
#    thêm vào cuối + xóa weights.npz để train lại từ đầu.
#
#  v3.2: Rút từ 15 class xuống 10 "cảm xúc đơn giản" — bỏ các class
#  trùng lặp (vui_nhon trùng happy) hoặc không phải cảm xúc thuần
#  (mysterious, adventurous, confident, grateful). Đổi tên class sang
#  tiếng Anh cho rõ nghĩa, nhãn hiển thị "vi" vẫn giữ tiếng Việt.
EMOTIONS = [
    "happy", "sad", "romantic", "energetic",
    "relaxed", "lonely", "stressed", "focused",
    "nostalgic", "angry",
]

# ── METADATA HIỂN THỊ ────────────────────────────────────────────
# valence  : Sắc thái cảm xúc, [-1 (tiêu cực) .. +1 (tích cực)]
# arousal  : Cường độ năng lượng, [-1 (thấp/êm) .. +1 (cao/dồn dập)]
# Theo mô hình GEMS + Circumplex of Affect (Russell) — xem thongtin.docx.
# "happy" gộp Vui vẻ (+0.9/+0.4) và Vui nhộn (+0.8/+0.6) → lấy trung bình.
# "angry" không có trong bảng gốc, suy theo circumplex: năng lượng cao,
# sắc thái tiêu cực hơn cả "stressed".
EMOTION_META = {
    "happy":     {"vi": "Vui vẻ",     "emoji": "😄", "valence":  0.85, "arousal":  0.50},
    "sad":       {"vi": "Buồn bã",    "emoji": "😢", "valence": -0.80, "arousal": -0.60},
    "romantic":  {"vi": "Lãng mạn",   "emoji": "🥰", "valence":  0.70, "arousal":  0.20},
    "energetic": {"vi": "Năng động",  "emoji": "⚡", "valence":  0.60, "arousal":  0.90},
    "relaxed":   {"vi": "Thư giãn",   "emoji": "🌿", "valence":  0.80, "arousal": -0.40},
    "lonely":    {"vi": "Cô đơn",     "emoji": "🌙", "valence": -0.70, "arousal": -0.50},
    "stressed":  {"vi": "Căng thẳng", "emoji": "🔥", "valence": -0.60, "arousal":  0.80},
    "focused":   {"vi": "Tập trung",  "emoji": "🎯", "valence":  0.40, "arousal": -0.50},
    "nostalgic": {"vi": "Hoài niệm",  "emoji": "🍂", "valence": -0.20, "arousal": -0.30},
    "angry":     {"vi": "Tức giận",   "emoji": "😡", "valence": -0.75, "arousal":  0.85},
}

# ── TỪ PHỦ ĐỊNH ──────────────────────────────────────────────────
# Nếu một từ/cụm trong tập này đứng ngay trước từ/cụm cảm xúc → điểm ×-0.6
# Ví dụ: "không vui" → happy bị trừ thay vì cộng
# Cụm 2-3 từ ("không hề", "chẳng bao giờ") cũng được nhận diện — không cần
# sửa code, cứ thêm cụm phủ định mới vào đây là rule_score() tự bắt được.
NEGATIONS = {
    # 1 từ
    "không", "chẳng", "chả", "đâu", "chưa", "khỏi", "ko", "k",
    "hổng",  # phương ngữ Nam: "hổng vui" = "không vui"
    # 2 từ
    "không hề", "chẳng hề", "chả hề", "chưa hề",
    "không phải", "chẳng phải", "chả phải",
    "không có", "đâu có", "có đâu", "đâu phải",
    "không được", "chẳng được", "chả được",  # modal: "không được vui cho lắm"
    "hổng có",  # phương ngữ Nam modal
    # 3 từ
    "không bao giờ", "chẳng bao giờ", "chả bao giờ", "chưa bao giờ",
}

# ── TỪ ĐIỂN CHÍNH ────────────────────────────────────────────────
LEXICON = {

    # ── HAPPY (Vui vẻ) ───────────────────────────────────────────
    "happy": {
        # Từ gốc
        "vui":2.5, "hạnh phúc":3.0, "sung sướng":2.5, "tuyệt":2.0, "tuyệt vời":3.0,
        "phấn khởi":2.5, "hào hứng":2.0, "thích":1.5, "cười":2.0, "mừng":2.5,
        "haha":2.5, "hehe":2.0, "tốt":1.5, "hay":1.5, "đỉnh":2.0, "khoái":2.5,
        "sướng":2.5, "tươi":2.0, "ok":1.2, "thú vị":2.0,
        "xinh":1.5, "đẹp":1.5, "vui quá":3.0, "thú vui":2.0, "rạng rỡ":2.5,
        # Mở rộng — slang & tiếng Anh thông dụng
        "happy":3.0, "good":2.0, "great":2.5, "awesome":3.0, "feel good":3.0,
        "vibe":2.0, "nice":2.0, "yay":2.5, "yeah":1.8, "woah":1.8,
        "oke":1.5, "okê":1.5, "ổn":1.5, "ổn áp":2.0, "xịn":2.5, "xịn xò":3.0,
        "phê":2.0, "phê quá":2.5, "đã":2.0, "đã quá":2.5,
        "mlem":1.8, "ngon":1.8, "ngon lành":2.0,
        "vui lắm":3.0, "vui ghê":3.0, "vui vl":3.0, "vui vcl":3.0,
        "trời đẹp":2.0, "ngày đẹp":2.5, "đẹp quá":2.5,
        "vui dữ":2.5, "đỉnh nóc":2.5, "max vui":2.5, "vui banh nóc":3.0,
        # Hài hước / vui nhộn (gộp từ class trùng "vui_nhon")
        "vui nhộn":3.0, "hài hước":3.0, "buồn cười":3.0, "tếu":2.5, "nhí nhố":3.0,
        "lầy lội":3.0, "troll":2.5, "hề":2.5, "comedy":3.0, "funny":3.0, "vui tính":2.5,
        "tinh nghịch":2.5, "nghịch ngợm":2.5, "ngộ nghĩnh":2.5, "đáng yêu":2.0,
        "cute":2.0, "tưng tửng":2.5, "quậy":2.5, "hớn hở":2.5, "nhộn nhịp":2.0,
        "vui vẻ hài":2.5, "cười bể bụng":3.0, "meme":2.5, "hài":2.5, "giải trí":2.0,
        "lol":3.0, "hihi":2.5, "lmao":3.0, "rofl":3.0,
        "joke":3.0, "humor":3.0, "sarcasm":2.0, "irony":2.0,
        "cười ngất":3.0, "cười vỡ bụng":3.0, "cười té ghế":3.0, "cười lộn ruột":3.0,
        "hài vl":3.0, "hài vcl":3.0, "bá đạo":2.5, "dở hơi":2.0, "điên điên":2.0,
        "nhố nhăng":2.5, "quậy phá":2.5, "trốn học":1.5, "nghịch":2.0,
        "hài nước mắt":2.5, "stand up":3.0, "roast":2.5,
        "gag":2.5, "prank":2.5, "sketch":2.0,
        "silly":2.5, "goofy":2.5, "playful":2.5, "witty":2.5,
        "bông đùa":2.5, "đùa nghịch":2.5, "giỡn":2.5, "chọc":2.0,
        # Trạng thái vui nội tâm (STATE, phân biệt với energetic=hành động)
        "tâm trạng tốt":3.0, "mood tốt":3.0, "vui bụng":2.5, "phấn chấn":2.5,
        "hài lòng":2.5, "toại nguyện":2.5, "thỏa mãn":2.5, "vui mừng":3.0,
        "mừng rỡ":3.0, "vui sướng":3.0, "sướng rơn":2.5, "nhẹ nhõm":2.5,
        "tươi tắn":2.5, "rạng ngời":2.5, "tươi vui":2.5, "nở nụ cười":2.5,
        "niềm vui":3.0, "vui mừng khôn xiết":3.0, "sung sướng":3.0,
        # Slang 2024-2025 & phương ngữ Nam
        "siu xịn mịn":3.0, "căng đét":2.5, "đỉnh của chóp":3.0,
        "siêu đỉnh":3.0, "vui hết nấc":3.0, "vui điên":2.5,
        "glow up":2.5, "slay":2.5, "vibes":2.0, "peak":2.5,
        "mắc cười":2.5, "dzui":2.5, "dui":2.0,
        "thích vl":3.0, "ngon vl":3.0, "đỉnh vl":3.0,
    },

    # ── SAD (Buồn bã) ────────────────────────────────────────────
    "sad": {
        # Từ gốc
        "buồn":3.0, "khóc":3.0, "đau":2.5, "khổ":2.5, "tủi":2.5, "thất vọng":2.5,
        "chán":2.0, "nản":2.0, "mệt mỏi":1.8, "đau lòng":3.0, "tan vỡ":3.0,
        "chia tay":3.0, "mất":1.0, "trống rỗng":2.5, "héo":2.0, "ủ rũ":2.5,
        "nước mắt":3.0, "tội nghiệp":1.5, "bi":2.0, "đau khổ":3.0,
        "tuyệt vọng":3.0, "vô vọng":2.5, "chán nản":2.5, "tiếc":1.0, "hối hận":2.0,
        # Mở rộng
        "sad":3.0, "cry":3.0, "depressed":3.0, "down":2.0, "low":1.8,
        "heartbreak":3.0, "broken":2.5, "unhappy":2.5, "grief":2.5,
        "buồn thiu":3.0, "buồn quá":3.0, "buồn vl":3.0, "buồn vcl":3.0,
        "chán đời":3.0, "chán lắm":2.5, "chán quá":2.5,
        "nặng lòng":2.5, "nặng nề":2.0, "ảm đạm":2.5,
        "thất bại":2.5, "bỏ cuộc":2.5, "thua":1.0,
        "mưa buồn":2.5, "đêm buồn":2.5, "cô quạnh":2.5,
        # Phủ định tu từ: "vui gì chứ" — từ chối cảm xúc tích cực → buồn
        "vui gì chứ":3.0, "vui gì đâu":3.0, "vui sao được":2.5,
        "vui gì nữa":3.0, "làm sao vui được":2.5, "vui được đâu":2.5,
        "hạnh phúc gì chứ":2.5, "vui lên được đâu":2.5,
        # Phủ định ngầm / bất lực / chán nản
        "thôi rồi":2.5, "đành vậy":2.0, "thế là xong":2.5,
        "biết sao giờ":2.0, "đời là vậy":1.5, "không còn thiết":2.5,
        # Buồn im lặng / nội tâm (quiet sadness — hay bị miss)
        "ngậm ngùi":3.0, "chạnh lòng":2.5, "u sầu":3.0, "u uẩn":2.5,
        "sụt sùi":3.0, "thổn thức":2.5, "nghẹn ngào":2.5, "nức nở":3.0,
        "cay mắt":2.5, "nghẹn lời":2.5, "đau đáu":2.5, "bứt rứt":2.0,
        # Mất động lực / thờ ơ (resignation sadness)
        "không còn thiết":2.5, "chẳng thiết gì":3.0, "không thiết gì nữa":3.0,
        "không muốn gặp ai":3.0, "buông xuôi":2.5, "mặc kệ tất cả":2.5,
        "chẳng muốn gì nữa":2.5, "không còn cảm giác":2.5,
        # Thất vọng sâu
        "vỡ mộng":3.0, "hụt hẫng":2.5, "chưng hửng":2.0,
        "thất vọng về bản thân":3.0, "không như kỳ vọng":2.5,
        # Slang
        "não nề":2.5, "quá não":2.0, "low mood":2.5,
    },

    # ── ROMANTIC (Lãng mạn) ──────────────────────────────────────
    "romantic": {
        # Từ gốc
        "yêu":3.0, "thương":2.5, "nhớ":2.0, "tim":2.0, "tình yêu":3.0,
        "lãng mạn":3.0, "ngọt ngào":2.5, "dịu dàng":2.5, "ôm":2.0, "hôn":2.5,
        "bạn gái":2.5, "bạn trai":2.5, "người yêu":3.0, "crush":3.0,
        "si mê":3.0, "say đắm":3.0, "ngưỡng mộ":2.0, "hẹn hò":3.0,
        "valentine":2.5, "ánh trăng":2.0, "hoa":1.5, "tình":2.0, "yêu thương":2.5,
        "nhớ nhung":2.5, "thầm thương":2.5, "mê":2.0,
        # Mở rộng
        "love":3.0, "romance":3.0, "darling":2.5, "sweetie":2.5, "baby":2.0,
        "miss you":3.0, "yêu em":3.0, "yêu anh":3.0, "yêu bạn":3.0,
        "thương nhau":3.0, "bên nhau":2.5, "cùng nhau":2.0,
        "nắm tay":2.5, "cạnh nhau":2.5, "ôm nhau":3.0,
        "tình cảm":2.5, "cảm mến":2.5, "đặc biệt":1.8,
        "ánh mắt":2.0, "nụ cười":2.0, "má đỏ":2.0,
        "thích bạn":3.0, "thích anh":3.0, "thích em":3.0,
    },

    # ── ENERGETIC (Năng động) ────────────────────────────────────
    "energetic": {
        # Từ gốc
        "năng động":3.0, "phấn khích":2.5, "hứng khởi":2.5, "sôi nổi":2.5,
        "nhiệt huyết":2.5, "party":3.0, "nhảy":2.5, "dance":2.5, "workout":3.0,
        "gym":3.0, "chạy":2.0, "tập":2.0, "thể thao":2.5, "festival":2.5,
        "nhạc mạnh":2.5, "bass":2.0, "beat":2.0, "remix":2.0, "hype":3.0,
        "lit":2.5, "siêu":2.0, "mạnh":1.5, "khỏe":2.0, "pump":2.5, "energetic":3.0,
        # Mở rộng
        "fire":2.5, "energy":2.5, "hype quá":3.0, "bùng cháy":3.0,
        "sôi động":3.0, "nhiệt tình":2.5, "cuồng nhiệt":3.0, "điên cuồng":2.5,
        "run":2.0, "jog":2.0, "cardio":2.5, "crossfit":2.5, "boxing":2.5,
        "bóng đá":2.0, "bóng rổ":2.0, "cầu lông":2.0,
        "nhảy nhót":2.5, "múa":2.0, "aerobic":2.5,
        "tiệc":2.5, "club":2.5, "bar":2.0, "rave":3.0,
        "let's go":2.5, "go go":2.5, "nào đi":2.0,
        "bừng tỉnh":2.5, "tràn đầy năng lượng":3.0,
    },

    # ── RELAXED (Thư giãn) ───────────────────────────────────────
    "relaxed": {
        # Từ gốc
        "thư giãn":3.0, "nghỉ ngơi":2.5, "bình yên":3.0, "nhẹ nhàng":2.5,
        "yên tĩnh":2.5, "tĩnh lặng":2.5, "lofi":3.0, "acoustic":2.5,
        "nhẹ":2.0, "dịu":2.0, "êm":2.0, "calm":3.0, "chill":2.5, "relax":3.0,
        "cozy":2.5, "ấm áp":2.5, "mưa":2.0, "trà":2.0, "cà phê":1.8,
        "đọc sách":2.0, "nằm":1.5, "yên ả":2.5, "comfortable":2.5, "thư thái":3.0,
        # Mở rộng
        "chill thôi":3.0, "chill vibe":3.0, "chill lắm":2.5,
        "relax thôi":3.0, "nghỉ xả hơi":2.5, "xả hơi":2.5,
        "slow down":2.5, "take it easy":2.5, "easy":2.0,
        "mellow":3.0, "soft":2.0, "gentle":2.5, "peaceful":3.0,
        "ngủ":1.5, "ngủ ngon":2.0, "buồn ngủ":1.5,
        "thiền":2.5, "meditation":2.5, "yoga":2.5,
        "cuối tuần":2.0, "nghỉ lễ":2.0, "không làm gì":2.0,
        "lười":1.5, "lười biếng":1.5, "nằm dài":2.0,
        "nhẹ lòng":2.5, "thanh thản":3.0, "bình thản":2.5,
        "gió":1.5, "nắng nhẹ":2.0, "buổi sáng yên tĩnh":2.5,
        "touch grass":2.0, "nạp lại năng lượng":2.5, "detox":2.0, "chữa lành":2.5,
        # Slang 2024-2025
        "chill phết":3.0, "max chill":3.0, "vibe chill":2.5,
        "chill mode":3.0, "no stress":2.5, "just vibes":2.5,
    },

    # ── LONELY (Cô đơn) ──────────────────────────────────────────
    "lonely": {
        # Từ gốc
        "cô đơn":3.0, "một mình":2.5, "lạc lõng":2.5, "xa":1.5, "nhớ nhà":3.0,
        "tha hương":2.5, "vắng":2.0, "trống":2.0, "im lặng":2.0, "đêm khuya":2.5,
        "đêm":1.5, "khuya":2.0, "tối":1.2, "bóng tối":2.0, "bơ vơ":3.0,
        "lẻ loi":3.0, "đơn độc":3.0, "không ai":2.5, "vắng vẻ":2.5,
        "nỗi niềm":2.0, "tâm tư":2.0, "thao thức":2.0, "xa nhà":3.0,
        # Mở rộng
        "lonely":3.0, "alone":2.5, "empty":2.5, "hollow":2.5, "isolated":3.0,
        "không có ai":3.0, "chẳng có ai":3.0, "không ai hiểu":3.0,
        "ngồi một mình":3.0, "đứng một mình":2.5, "đi một mình":2.5,
        "phòng trống":2.5, "nhà trống":2.5, "vắng lặng":2.5,
        "xa cách":2.5, "cách xa":2.5, "ở xa":2.0,
        "nhìn mưa":2.0, "nhìn cửa sổ":2.0, "ngồi nhìn":1.8,
        "tâm sự":2.0, "không ai tâm sự":3.0, "giữ trong lòng":2.5,
        "buổi tối":1.5, "đêm tối":2.0, "nửa đêm":2.5,
        # Phương ngữ Nam + biến thể
        "hổng có ai":3.0, "thui thủi":2.5, "lủi thủi":2.5,
        "chẳng có ai quan tâm":3.0, "chẳng ai hỏi thăm":3.0,
    },

    # ── STRESSED (Căng thẳng) ────────────────────────────────────
    "stressed": {
        # Từ gốc
        "căng thẳng":3.0, "áp lực":3.0, "stress":3.0, "lo lắng":2.5, "lo":2.0,
        "sợ":2.0, "hồi hộp":2.0, "bất an":2.5, "bực":2.5,
        "khó chịu":2.0, "bực mình":2.5, "chán ghét":2.0, "ghét":2.0,
        "mệt":1.5, "kiệt sức":2.5, "burn out":3.0, "quá tải":3.0, "deadline":2.5,
        "thi":1.5, "không ổn":2.5, "tệ":2.0, "khủng hoảng":3.0, "rắc rối":2.0,
        # Mở rộng
        "stressed":3.0, "anxious":3.0, "panic":3.0, "overwhelmed":3.0,
        "exhausted":2.5, "burnout":3.0, "toxic":2.5, "pressure":3.0,
        "lo sợ":2.5, "hoảng loạn":3.0, "hoảng sợ":3.0,
        "mệt nhoài":2.5, "mệt xỉu":2.5, "kiệt":2.0, "đuối":2.0,
        "chịu không nổi":3.0, "không chịu được":3.0, "quá sức":3.0,
        "stress nặng":3.0, "áp lực nặng":3.0, "căng quá":3.0,
        "khó thở":2.5, "nghẹt thở":2.5, "nặng nề":2.0,
        "thi cử":2.0, "ôn thi":2.0, "sắp thi":2.5,
        "tranh cãi":2.0, "mâu thuẫn":2.5, "xung đột":2.5,
        "quá tải tinh thần":3.0, "không thở được":2.5, "ngộp thở":2.5,
        # Slang 2024-2025
        "toxic vl":3.0, "toxic quá":2.5, "drama quá":2.5, "drama vl":3.0,
        "overthink":2.5, "overthinking":2.5, "ngộp lắm":2.5, "ngộp quá":2.5,
        "rối não":2.5, "rối bời":2.5,
    },

    # ── FOCUSED (Tập trung) ──────────────────────────────────────
    "focused": {
        # Từ gốc
        "tập trung":3.0, "học":2.0, "làm việc":2.5, "study":2.5, "work":2.0,
        "focus":3.0, "nghiên cứu":2.5, "đọc":1.5, "viết":1.5, "code":2.5,
        "lập trình":2.5, "dự án":2.0, "project":2.0, "deadline":1.5, "hiệu quả":2.5,
        "năng suất":3.0, "productive":3.0, "deep work":3.0, "sáng tạo":2.0,
        "brainstorm":2.5, "nhiệm vụ":2.0, "task":2.0, "mục tiêu":2.0, "kế hoạch":2.0,
        # Mở rộng
        "concentrate":3.0, "productivity":3.0, "workflow":2.5,
        "học bài":2.5, "ôn bài":2.5, "làm bài":2.5, "làm đề":2.5,
        "debug":2.5, "coding":2.5, "programming":2.5, "develop":2.0,
        "thiết kế":2.0, "design":2.0, "vẽ":1.5, "phác thảo":2.0,
        "meeting":2.0, "họp":2.0, "thuyết trình":2.5, "báo cáo":2.0,
        "lo làm":2.0, "đang làm":1.5, "đang học":2.0, "đang code":2.5,
        "không làm phiền":2.5, "im lặng để làm":2.5,
        "cần nhạc nền":2.0, "nhạc tập trung":3.0, "nhạc học bài":3.0,
    },

    # ── NOSTALGIC (Hoài niệm) ────────────────────────────────────
    "nostalgic": {
        # Từ gốc
        "hoài niệm":3.0, "nhớ về":2.5, "kỷ niệm":3.0, "ngày xưa":3.0, "tuổi thơ":3.0,
        "quá khứ":2.5, "hồi đó":2.5, "thuở":2.0, "xưa cũ":2.5, "cũ":1.5, "hồi ức":3.0,
        "thời gian":1.5, "năm tháng":2.5, "nhớ lại":2.5, "vintage":2.5, "retro":3.0,
        "bồi hồi":3.0, "bâng khuâng":2.5, "luyến tiếc":2.5, "thanh xuân":3.0,
        "tuổi học trò":2.5, "ngày ấy":2.5, "hoài cổ":3.0, "nuối tiếc":2.5, "ngày tháng cũ":3.0,
        # Mở rộng
        "nostalgia":3.0, "throwback":3.0, "old days":2.5, "memories":3.0,
        "nhớ ngày xưa":3.0, "nhớ hồi":2.5, "nhớ lúc":2.5,
        "hồi nhỏ":3.0, "hồi bé":3.0, "lúc nhỏ":3.0, "lúc bé":3.0,
        "ký ức":3.0, "ký ức tuổi thơ":3.0,
        "trường cũ":2.5, "bạn cũ":2.5, "thầy cô cũ":2.5, "lớp học cũ":2.5,
        "nhạc xưa":3.0, "bài hát cũ":3.0, "nhạc cũ":3.0,
        "album ảnh":2.5, "ảnh cũ":2.5, "ảnh xưa":2.5,
        "mùa hè năm đó":3.0, "mùa xuân năm xưa":2.5,
        "vẫn nhớ":2.5, "không quên":2.0, "mãi nhớ":2.5,
        "chạnh lòng":2.5, "bồn chồn":2.0,
    },

    # ── ANGRY (Tức giận) ─────────────────────────────────────────
    "angry": {
        # Từ gốc
        "tức giận":3.0, "phẫn nộ":3.0, "giận dữ":3.0, "nổi giận":3.0, "nổi điên":3.0,
        "điên tiết":3.0, "bực tức":2.5, "căm phẫn":3.0, "bất bình":2.5, "hận":2.5,
        "thù":2.0, "ghét cay ghét đắng":3.0, "nóng máu":2.5, "sôi máu":3.0,
        "mất bình tĩnh":2.5, "không thể chấp nhận được":3.0, "quá đáng":2.5,
        "vô lý":2.0, "ức chế":2.5, "uất":2.5, "cáu tiết":2.5,
        # Mở rộng — slang & tiếng Anh thông dụng
        "angry":3.0, "furious":3.0, "rage":3.0, "mad":2.5, "pissed":2.5,
        "pissed off":3.0, "outrage":2.5, "fury":3.0, "hate this":2.5,
        "tức điên người":3.0, "giận tím người":3.0, "không thể tin được":2.5,
        "quá vô lý":2.5, "muốn đập phá":3.0, "muốn hét lên":2.5,
        "chịu hết nổi":2.5, "drama gì thế":2.0, "cạn lời":2.0,
        "ai cho phép":2.5, "thái độ":2.0, "xúc phạm":2.5, "khinh thường":2.5,
        "không ngờ":2.0, "thật không ngờ":2.5, "bực bội":2.0,
        "bất công":2.5, "phản bội":2.5,
        # Tức vì bị đối xử bất công (directed anger — phân biệt với stressed)
        "ăn hiếp":2.5, "bị chà đạp":3.0, "bị ức hiếp":3.0,
        "không thể tha thứ":2.5, "quá đáng":2.5, "thật quá đáng":3.0,
        "tức quá":3.0, "giận sôi người":3.0, "tức không nói được":2.5,
        "bị đối xử tệ":3.0, "làm tôi tức":3.0, "khiến tôi tức giận":3.0,
        "uất ức quá":3.0, "căm ghét":3.0, "tức vô cùng":3.0,
        # Giận dữ rõ ràng (slang)
        "tức vl":3.0, "giận vl":3.0, "bực vl":3.0,
        "điên tiết vl":3.0, "tức điên":3.0, "nổi điên vl":3.0,
    },
}
