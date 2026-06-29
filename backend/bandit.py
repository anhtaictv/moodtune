"""
MoodTune RLUF - Thompson Sampling Multi-Armed Bandit (v3.0)
============================================================
Tối ưu Hybrid Playlist Mixer bằng Reinforcement Learning from User Feedback,
100% NumPy thuần (Beta-Bernoulli Thompson Sampling).

- Mỗi cảm xúc có 2 "bandit" độc lập:
  1. source bandit: 2 arm ("online" Jamendo / "local" Kho nhạc) - quyết định
     tỉ lệ trộn nhạc Online:Offline.
  2. tags bandit: N arm (các cặp tag trong EMOTION_TAGS[emotion]) - quyết định
     thẻ tag Jamendo nào hợp gu người dùng nhất.

- Mỗi arm có tham số Beta(a, b), khởi tạo a=b=1 (uniform prior).
- Reward: Like (+1) -> a += 1; Dislike/Next (-1) -> b += 1.
- sample_*: lấy mẫu theta ~ Beta(a,b) cho từng arm (np.random.beta) ->
  arm có theta lớn nhất / tỉ lệ theta được chọn (Thompson Sampling).
- State lưu ra JSON, lazy-init khi gặp emotion/arm mới, theo cùng pattern
  với dynamic_vocab.json / weights_meta.json.
"""

import json
import os
import time
import numpy as np

BANDIT_SAVE_INTERVAL = 30  # seconds between deferred saves


class ThompsonBandit:
    def __init__(self, state_path):
        self.state_path = state_path
        self.state = {"source": {}, "tags": {}}
        self._dirty = False
        self._last_save = 0.0
        self._load()

    def _load(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, encoding="utf-8") as f:
                    data = json.load(f)
                    self.state["source"] = data.get("source", {})
                    self.state["tags"] = data.get("tags", {})
            except Exception:
                pass

    def save(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False)
        os.replace(tmp, self.state_path)
        self._dirty = False
        self._last_save = time.time()

    def _maybe_save(self):
        if self._dirty and time.time() - self._last_save >= BANDIT_SAVE_INTERVAL:
            self.save()

    # ─── SOURCE BANDIT (Online vs Local) ─────────────────────────
    def _source_arms(self, emotion):
        arms = self.state["source"].setdefault(emotion, {})
        arms.setdefault("online", [1, 1])
        arms.setdefault("local", [1, 1])
        return arms

    def sample_mix(self, emotion, total=12):
        """Lấy mẫu Beta cho 2 arm online/local, chia `total` theo tỉ lệ
        theta_online / (theta_online + theta_local). Trả (n_online, n_local)."""
        arms = self._source_arms(emotion)
        a_on, b_on = arms["online"]
        a_lo, b_lo = arms["local"]
        theta_on = np.random.beta(a_on, b_on)
        theta_lo = np.random.beta(a_lo, b_lo)
        total_theta = theta_on + theta_lo
        if total_theta <= 0:
            ratio_on = 0.5
        else:
            ratio_on = theta_on / total_theta
        n_online = int(round(total * ratio_on))
        n_online = max(0, min(total, n_online))
        n_local = total - n_online
        return n_online, n_local

    def update_source(self, emotion, source, reward):
        """reward = +1 (like) hoặc -1 (dislike/next)."""
        if source not in ("online", "local"):
            return
        arms = self._source_arms(emotion)
        a, b = arms[source]
        if reward > 0:
            a += 1
        else:
            b += 1
        arms[source] = [a, b]
        self._dirty = True
        self._maybe_save()

    # ─── TAGS BANDIT (chọn cặp tag Jamendo) ──────────────────────
    def _tag_arms(self, emotion, n_options):
        arms = self.state["tags"].setdefault(emotion, {})
        for i in range(n_options):
            arms.setdefault(str(i), [1, 1])
        return arms

    def sample_tag_index(self, emotion, n_options):
        """Lấy mẫu Beta cho từng arm (cặp tag), trả về index của arm có
        theta lớn nhất (argmax Thompson Sampling)."""
        if n_options <= 0:
            return 0
        arms = self._tag_arms(emotion, n_options)
        thetas = [np.random.beta(*arms[str(i)]) for i in range(n_options)]
        return int(np.argmax(thetas))

    def update_tag(self, emotion, tag_idx, reward):
        arms = self.state["tags"].setdefault(emotion, {})
        a, b = arms.get(str(tag_idx), [1, 1])
        if reward > 0:
            a += 1
        else:
            b += 1
        arms[str(tag_idx)] = [a, b]
        self._dirty = True
        self._maybe_save()

    # ─── SUMMARY (hiển thị "gu của bạn" trên UI) ─────────────────
    def get_summary(self, emotion):
        """Tỉ lệ kỳ vọng (mean Beta = a/(a+b)) cho online/local, KHÔNG sample
        - dùng để hiển thị ổn định trên UI."""
        arms = self._source_arms(emotion)
        a_on, b_on = arms["online"]
        a_lo, b_lo = arms["local"]
        m_on = a_on / (a_on + b_on)
        m_lo = a_lo / (a_lo + b_lo)
        total = m_on + m_lo
        if total <= 0:
            ratio_on, ratio_lo = 0.5, 0.5
        else:
            ratio_on, ratio_lo = m_on / total, m_lo / total
        return {"online": round(ratio_on, 3), "local": round(ratio_lo, 3)}
