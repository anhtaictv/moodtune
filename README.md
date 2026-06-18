[🇻🇳 Bản tiếng Việt](README.vi.md)

# MoodTune

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
