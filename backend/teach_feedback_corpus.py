"""
Mở rộng pretrain corpus bằng dữ liệu feedback thật từ feedback_log.jsonl.

Vì sao cần script này: SEED_DATA trong emotion_mlp.py chỉ có ~88 câu viết
tay, nhưng feedback_log.jsonl đã tích lũy hàng nghìn cặp (câu, cảm xúc
đúng) do người dùng thật sửa qua thời gian. Replay buffer production lại
bị cắt cứng ở 500 mẫu gần nhất (xem emotion_mlp.py _save_replay), nên phần
lớn lịch sử feedback không còn được dùng để train nữa - chỉ nằm chết trong
log. Script này gom lại toàn bộ feedback hợp lệ (loại bỏ nhãn cũ kiểu Việt
hoá như "vui_ve", "bi_an" từ version trước không còn khớp EMOTIONS hiện
tại), trộn với SEED_DATA + cụm cảm xúc dài trong LEXICON, giữ lại một phần
holdout (stratified theo nhãn) để đo accuracy thật trước/sau - không chỉ
nhìn vài câu demo - rồi pretrain lại.

Cách dùng AN TOÀN (khuyến nghị) - chạy thử trên 1 bản COPY của weights
production, không đụng tới model đang chạy thật:

    cd backend
    cp weights.npz weights_feedback_test.npz
    cp weights_meta.json weights_feedback_test_meta.json
    cp weights_replay.json weights_feedback_test_replay.json
    python teach_feedback_corpus.py --weights weights_feedback_test

Chỉ sau khi xem accuracy holdout before/after và thấy ổn, mới áp dụng vào
production thật (cần dừng/restart service vì AttentionMLP nạp weights.npz
lúc khởi động, xem README.md "Running the backend"):

    python teach_feedback_corpus.py --weights weights

Tham số:
    --weights PATH      prefix bộ weights cần dạy (PATH.npz, PATH_meta.json,
                         PATH_replay.json). Mặc định "weights_feedback_test"
                         để KHÔNG vô tình đụng vào production nếu quên cờ.
    --epochs N           số epoch pretrain (mặc định 400, giống cold-start
                         trong EmotionEngine.__init__).
    --holdout-frac F    tỉ lệ giữ lại làm test set, chia stratified theo
                         từng nhãn (mặc định 0.15).
    --seed N             random seed cho việc chia train/holdout (mặc định 42).
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

from emotion_mlp import (EmotionEngine, SEED_DATA, NEGATION_AUGMENT_DATA, ANGER_AUGMENT_DATA,
                          SAD_AUGMENT_DATA, ROMANTIC_AUGMENT_DATA, EMOTIONS, EMOTION_META, LEXICON, to_token_ids)

LOG_PATH = os.path.join(os.path.dirname(__file__), "feedback_log.jsonl")


def load_feedback_pairs(path=LOG_PATH):
    """Đọc feedback_log.jsonl, trả về list (text, label_idx) duy nhất.
    Chỉ lấy entry type=feedback có "correct" nằm trong EMOTIONS hiện tại -
    tự động loại các nhãn taxonomy cũ (vd "vui_ve", "bi_an", "phieu_luu")
    còn sót lại từ version trước khi đổi sang tên tiếng Anh."""
    seen = {}
    skipped_old_taxonomy = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") != "feedback":
                continue
            correct = d.get("correct")
            text = (d.get("text") or "").strip()
            if not text:
                continue
            if correct not in EMOTIONS:
                skipped_old_taxonomy += 1
                continue
            seen[text] = EMOTIONS.index(correct)  # trùng câu -> giữ nhãn mới nhất
    print(f"[teach_feedback_corpus] Bỏ qua {skipped_old_taxonomy} dòng nhãn taxonomy cũ "
          f"(không khớp EMOTIONS hiện tại).")
    return list(seen.items())


def collect_dead_phrase_examples():
    """Giống teach_model3.py: 1 mẫu trực tiếp cho mọi cụm cảm xúc >=3 từ
    trong LEXICON, đảm bảo embedding của các cụm dài cũng được ôn lại."""
    examples = []
    for emo, words in LEXICON.items():
        label = EMOTIONS.index(emo)
        for phrase in words:
            if len(phrase.split()) >= 3:
                examples.append((phrase, label))
    return examples


def stratified_split(pairs, holdout_frac, seed):
    """Chia train/holdout theo từng nhãn riêng để mỗi cảm xúc đều có mặt
    cân đối ở holdout, không bị lệch vì nhãn ít mẫu (vd "sad" chỉ có 208)."""
    rng = np.random.RandomState(seed)
    by_label = collections.defaultdict(list)
    for text, label in pairs:
        by_label[label].append((text, label))
    train, holdout = [], []
    for label, items in by_label.items():
        items = list(items)
        rng.shuffle(items)
        k = max(1, int(len(items) * holdout_frac))
        holdout.extend(items[:k])
        train.extend(items[k:])
    return train, holdout


def evaluate(engine, data):
    """Trả về (mlp_accuracy, hybrid_accuracy) trên tập data."""
    mlp_correct = 0
    hybrid_correct = 0
    for text, label in data:
        ids = to_token_ids(text)
        p_mlp = engine.mlp.predict(ids)
        if int(np.argmax(p_mlp)) == label:
            mlp_correct += 1
        out = engine.predict(text)
        if out["emotion"] == EMOTIONS[label]:
            hybrid_correct += 1
    n = max(len(data), 1)
    return mlp_correct / n, hybrid_correct / n


def per_class_report(engine, data):
    counts = collections.Counter()
    correct = collections.Counter()
    for text, label in data:
        out = engine.predict(text)
        counts[label] += 1
        if out["emotion"] == EMOTIONS[label]:
            correct[label] += 1
    print(f"{'Cảm xúc':12} {'N':>5} {'Accuracy':>10}")
    for label in range(len(EMOTIONS)):
        n = counts.get(label, 0)
        if n == 0:
            continue
        acc = correct.get(label, 0) / n
        vi = EMOTION_META[EMOTIONS[label]]["vi"]
        print(f"{EMOTIONS[label]:12} {n:>5} {acc*100:>9.1f}%   ({vi})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="weights_feedback_test",
                         help="prefix bộ weights cần dạy (mặc định: weights_feedback_test, KHÔNG phải production)")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"[teach_feedback_corpus] Nạp engine từ '{args.weights}'...")
    engine = EmotionEngine(args.weights)
    print(f"[teach_feedback_corpus] feedback_count={engine.feedback_count} | alpha={engine.alpha:.3f} "
          f"| vocab_size={engine.vocab_size} | replay={len(engine.replay)} mẫu")

    feedback_pairs = load_feedback_pairs()
    print(f"[teach_feedback_corpus] Đọc được {len(feedback_pairs)} cặp (câu, nhãn) duy nhất từ feedback_log.jsonl")

    train_feedback, holdout = stratified_split(feedback_pairs, args.holdout_frac, args.seed)
    print(f"[teach_feedback_corpus] Chia stratified: {len(train_feedback)} train / {len(holdout)} holdout "
          f"(holdout_frac={args.holdout_frac})")

    dead_phrases = collect_dead_phrase_examples()
    train_data = SEED_DATA + dead_phrases + NEGATION_AUGMENT_DATA + ANGER_AUGMENT_DATA + SAD_AUGMENT_DATA + ROMANTIC_AUGMENT_DATA + [(t, l) for t, l in train_feedback]
    print(f"[teach_feedback_corpus] Tổng corpus train: {len(train_data)} mẫu "
          f"(SEED_DATA={len(SEED_DATA)}, lexicon-phrase={len(dead_phrases)}, "
          f"negation-augment={len(NEGATION_AUGMENT_DATA)}, anger-augment={len(ANGER_AUGMENT_DATA)}, "
          f"sad-augment={len(SAD_AUGMENT_DATA)}, romantic-augment={len(ROMANTIC_AUGMENT_DATA)}, "
          f"feedback={len(train_feedback)})")

    print("\n[teach_feedback_corpus] Accuracy TRƯỚC khi train (trên holdout):")
    mlp_acc_before, hybrid_acc_before = evaluate(engine, holdout)
    print(f"  MLP-only: {mlp_acc_before*100:.1f}% | Hybrid: {hybrid_acc_before*100:.1f}%")

    print(f"\n[teach_feedback_corpus] Pretrain trên {len(train_data)} mẫu, {args.epochs} epochs "
          f"(có thể mất vài phút vì corpus lớn hơn nhiều so với teach_model3.py)...")
    engine.pretrain(train_data, epochs=args.epochs)

    print("\n[teach_feedback_corpus] Accuracy SAU khi train (trên holdout):")
    mlp_acc_after, hybrid_acc_after = evaluate(engine, holdout)
    print(f"  MLP-only: {mlp_acc_after*100:.1f}% | Hybrid: {hybrid_acc_after*100:.1f}%")

    print("\n[teach_feedback_corpus] Breakdown theo từng cảm xúc (Hybrid, sau khi train):")
    per_class_report(engine, holdout)

    print(f"\n[teach_feedback_corpus] Đã lưu lại weights tại '{args.weights}.npz' / "
          f"'{args.weights}_meta.json' / '{args.weights}_replay.json'.")
    print(f"[teach_feedback_corpus] So sánh: MLP {mlp_acc_before*100:.1f}% -> {mlp_acc_after*100:.1f}% | "
          f"Hybrid {hybrid_acc_before*100:.1f}% -> {hybrid_acc_after*100:.1f}%")


if __name__ == "__main__":
    sys.exit(main())
