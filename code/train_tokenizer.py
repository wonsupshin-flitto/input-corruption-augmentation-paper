#!/usr/bin/env python3
"""Train a SentencePiece multilingual tokenizer.

Spec (fixed by design): Unigram, byte_fallback, character_coverage 0.9995 (CJK),
special tokens (unk/bos/eos/pad). vocab is an argument (48k default, 64k comparison).

Usage:
  python train_tokenizer.py --input combined.txt --model-prefix tok_48k --vocab 48000
"""
import argparse
import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model-prefix", required=True)
    ap.add_argument("--vocab", type=int, default=48000)
    ap.add_argument("--model-type", default="unigram", choices=["unigram", "bpe"])
    ap.add_argument("--character-coverage", type=float, default=0.9995)
    ap.add_argument("--input-sentence-size", type=int, default=5_000_000)
    ap.add_argument("--num-threads", type=int, default=8)
    args = ap.parse_args()

    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        byte_fallback=True,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True,
        num_threads=args.num_threads,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>", pad_piece="<pad>",
        train_extremely_large_corpus=False,
    )
    print(f"done: {args.model_prefix}.model / .vocab (vocab={args.vocab})")


if __name__ == "__main__":
    main()
