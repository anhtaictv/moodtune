[🇻🇳 Bản tiếng Việt](README.vi.md)

# MoodTune

**Current version: `v3.8`** — see [Version history](#version-history) below for the full changelog.

Mood-based music recommendation: the user types a piece of text, the backend uses an AI engine (rule-based lexicon + a self-attention MLP that learns online from feedback) to infer the emotion, then looks up matching tracks via the Jamendo API (free, no login required).

## Structure

- `backend/` — Flask API (Python). Emotion engine (`emotion_mlp.py`, `lexicon.py`), a Thompson Sampling bandit that learns music taste (`bandit.py`), Jamendo track lookup + audio feature analysis (`audio_features.py`), entrypoint `app.py`.
- `frontend/` — single-page app (plain HTML/CSS/JS, no build step) in `index.html`.

## Running the backend

Requires Python 3.x.

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The backend runs at `http://localhost:5005`, with API base path `/api` (check `/api/health` to verify).

Environment variables (all have defaults, none are required for local runs):

| Variable | Default | Meaning |
| --- | --- | --- |
| `MOODTUNE_SECRET_KEY` | `moodtune_secret_2024` | Flask `secret_key`. Set your own value for production deployments. |
| `MOODTUNE_FRONTEND` | `https://anhtaictv.me` | Frontend origin, used to configure CORS. |
| `JAMENDO_CLIENT_ID` | `cf31dbfd` | Client ID used to call the Jamendo API. |

To run with pm2 (production), use the provided config:

```bash
pm2 start backend/ecosystem.config.js
```

## Running the frontend

Open `frontend/index.html` directly in a browser (or serve it via any static server, e.g. `http://localhost:5500`). When running on `localhost`/`file://`, the frontend automatically calls the backend directly at `http://localhost:5005/api` — no extra configuration needed.

In production, the frontend calls `/api` (a relative path), so you need a reverse proxy (IIS/Nginx) routing `/api` to the Flask backend. See `frontend/web.config` for a sample IIS configuration.

## Model data

`weights.npz`, `weights_meta.json`, `weights_replay.json`, and `dynamic_vocab.json` in `backend/` hold the model's learned state (vocabulary, MLP weights, replay buffer for online learning). `feedback_log.jsonl` stores the history of predictions/feedback/listening behavior, used by the bandit to learn music taste and by the `/api/stats` endpoint.

## Version history

Also viewable in the app itself by clicking the version tag next to the logo. Each `vX.X` report below has the full writeup (architecture diagrams, before/after comparisons, test results).

| Version | Name | Highlights |
| --- | --- | --- |
| `v3.8` | Multi-theme UI | Added an in-app theme picker with 3 themes: `Tối · Midnight` (default), `Sáng · Aurora` (light, slow-drifting pastel-mist background, auto-disables the animation under reduced-motion, shadow-elevated cards) and `Rực rỡ · Sunset` (slow-drifting sunset gradient, frosted-glass cards) — applies instantly, remembered via `localStorage`. Removed the gradient-text anti-pattern on the logo/headings, re-tokenized colors so every theme hits WCAG AA contrast, and the AI knowledge-graph canvas now reads its beam/bubble colors from the active theme instead of hardcoded green/violet. ([report](BaoCao_MoodTune_v3.8.md)) |
| `v3.7` | Production reliability pass | Added the app icon/favicon/PWA manifest and a presence widget (online/listening/total visits — real counts straight from active sessions); re-enabled the audio feature engine (fixed a `tempo`-as-array bug from a newer `librosa`) and switched from the Flask dev server to `waitress`; auto-heals an Embedding/VOCAB size mismatch after restarts; added rate-limiting + an admin-key bypass for `/api/learn`. No architecture change, fully backward-compatible with existing weights. ([report](BaoCao_MoodTune_v3.7.md)) |
| `v3.6` | Vietnamese negation & phrase fixes | Fixed the rule scorer: negation wasn't checked inside bigrams, and only single-word negations (`không`, `chẳng`...) were recognized — multi-word forms like `"không hề"`, `"chẳng bao giờ"` slipped through. No architecture change, fully backward-compatible with existing weights. ([report](BaoCao_MoodTune_v3.6.md)) |
| `v3.5` | Valence-Arousal normalization | Reduced 15 → 10 emotion classes to align with the Valence-Arousal (GEMS/Circumplex) model; renamed internal emotion keys to English (`vui_ve` → `happy`, etc.). ([report](BaoCao_MoodTune_v3.5.md)) |
| `v3.1` | New emotions | Added 3 emotions — confident 💪, grateful 🙏, angry 😡 — bringing the total from 12 to 15 classes. ([report](BaoCao_MoodTune_v3.1.md)) |
| `v3.0` | RLUF bandit & knowledge graph | Added a Thompson Sampling multi-armed bandit (pure NumPy, `bandit.py`) that learns music taste from likes/dislikes/skips, plus an interactive emotion knowledge graph (Canvas 2D) driven by the model's attention weights. ([report](BaoCao_MoodTune_v3.0.md)) |
| `v2.5` | Adaptive learning | Switched ReLU → Leaky ReLU in the hidden layer (fixes dying neurons under continuous online learning) and added adaptive L2 regularization that grows with feedback count; added the in-app changelog modal. ([report](BaoCao_MoodTune_v2.5.md)) |
| `v2.0` | Self-attention | Replaced bag-of-words input with an embedding layer + hand-written self-attention (Q,K,V) in NumPy, so word order (and negation) is preserved. Added dynamic vocab/weight expansion, an audio feature engine (BPM/spectral centroid/MFCC via librosa) to re-rank tracks, and a local-library hybrid online/offline mode. ([report](BaoCao_MoodTune_v2.0.md)) |
| `v1.1` | UI polish | Added the version badge in the UI; completed personalized recommendations, time-of-day suggestions, and analysis history. ([report](BaoCao_MoodTune_v1.1.md)) |
| `v1.0` | Foundation | Rule scorer + MLP hybrid engine (pure NumPy, no ML frameworks), Jamendo API integration, basic feedback/online learning. |

Two more documents cover the system in breadth rather than version-by-version: `BaoCao_MoodTune_TongQuan.md` (overall architecture) and `BaoCao_MoodTune_DacTrung_AIEngine_API.md` (AI engine & API deep dive).
