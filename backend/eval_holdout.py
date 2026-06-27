import sys, json, collections
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from emotion_mlp import EmotionEngine, EMOTIONS

engine = EmotionEngine('weights')

seen = {}
with open('feedback_log.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except: continue
        if d.get('type') != 'feedback': continue
        correct = d.get('correct')
        text = (d.get('text') or '').strip()
        if not text or correct not in EMOTIONS: continue
        seen[text] = EMOTIONS.index(correct)
pairs = list(seen.items())

rng = np.random.RandomState(42)
by_label = collections.defaultdict(list)
for text, label in pairs:
    by_label[label].append((text, label))
holdout = []
for label, items in by_label.items():
    items = list(items); rng.shuffle(items)
    k = max(1, int(len(items) * 0.15))
    holdout.extend(items[:k])

counts = collections.Counter()
correct_h = collections.Counter()
for text, label in holdout:
    out = engine.predict(text)
    counts[label] += 1
    if out['emotion'] == EMOTIONS[label]:
        correct_h[label] += 1

print(f'Holdout: {len(holdout)} mau')
print(f'{"Emotion":12} {"N":>5} {"Acc":>9}')
total_c = total_n = 0
for label in range(len(EMOTIONS)):
    n = counts.get(label, 0)
    if n == 0: continue
    c = correct_h.get(label, 0)
    acc = c / n
    total_c += c; total_n += n
    flag = ' <-- YEU' if acc < 0.75 else ''
    from lexicon import EMOTION_META
    vi = EMOTION_META[EMOTIONS[label]]['vi']
    print(f'{EMOTIONS[label]:12} {n:>5} {acc*100:>8.1f}%   ({vi}){flag}')
print(f'{"TOTAL":12} {total_n:>5} {total_c/total_n*100:>8.1f}%')
