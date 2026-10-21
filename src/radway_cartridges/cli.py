import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from .core import (Cartridge, MODELS, decode, distillation_loss, fingerprint,
                   load_model, prompt_ids, response_logits, tensor)

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def document(path):
    text = Path(path).read_text()
    if not text.strip():
        raise ValueError("Document is empty")
    return text

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def synthesize(args, model, tokenizer, model_id):
    text = document(args.document)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokenizer.decode(tokens[i:i + args.chunk_tokens])
              for i in range(0, len(tokens), args.chunk_tokens)]
    if args.samples < 1 or args.turns < 1:
        raise ValueError("samples and turns must be positive")
    seeds = ["Ask a specific factual question", "Ask for a summary",
             "Ask a question connecting two facts", "Ask for a structured list"]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        for sample in range(args.samples):
            context = chunks[sample % len(chunks)]
            history = []
            for turn in range(args.turns):
                instruction = seeds[(sample + turn) % len(seeds)] + " about the reference. Output only the question."
                question_tokens = decode(model, tokenizer, prompt_ids(
                    tokenizer, history + [{"role": "user", "content": instruction}], context),
                    max_new_tokens=args.question_tokens, sample=True)
                question = tokenizer.decode(question_tokens, skip_special_tokens=True).strip()
                if not question:
                    raise ValueError("Question generator produced an empty question")
                history.append({"role": "user", "content": question})
                teacher_prompt = prompt_ids(tokenizer, history, context)
                student_prompt = prompt_ids(tokenizer, history)
                response = decode(model, tokenizer, teacher_prompt, args.answer_tokens, sample=True)
                row = dict(model_id=model_id, model_revision=getattr(model.config, "_commit_hash", None),
                           config_hash=fingerprint(model.config), corpus_hash=digest(text),
                           sample=sample, turn=turn, question=question,
                           teacher_prompt=teacher_prompt, student_prompt=student_prompt,
                           response=response, answer=tokenizer.decode(response, skip_special_tokens=True))
                f.write(json.dumps(row) + "\n")
                f.flush()
                history.append({"role": "assistant", "content": row["answer"]})
            print(f"synthesized conversation {sample + 1}/{args.samples}", flush=True)

def train(args, model, tokenizer, model_id):
    text = document(args.document)
    rows = read_jsonl(args.data)
    if not rows or args.steps < 1 or args.cartridge_tokens < 1:
        raise ValueError("Nonempty training data, positive steps and cartridge length required")
    for row in rows:
        if (row["model_id"] != model_id or row["config_hash"] != fingerprint(model.config)
                or row["model_revision"] != getattr(model.config, "_commit_hash", None)
                or row["corpus_hash"] != digest(text)):
            raise ValueError("Training data model/revision/corpus mismatch; regenerate synthesis")
    ids = tokenizer.encode(text, add_special_tokens=True)[:args.cartridge_tokens]
    if len(ids) != args.cartridge_tokens:
        raise ValueError("Document is shorter than cartridge_tokens; choose a smaller cartridge")
    cartridge = Cartridge.initialize(model, tensor(ids, model))
    optimizer = torch.optim.AdamW(cartridge.parameters(), lr=args.lr, weight_decay=0.0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    random.shuffle(rows)
    losses = []
    for step in range(args.steps):
        row = rows[step % len(rows)]
        with torch.no_grad():
            teacher = response_logits(model, row["teacher_prompt"], row["response"])
        optimizer.zero_grad(set_to_none=True)
        student = response_logits(model, row["student_prompt"], row["response"], cartridge)
        loss = distillation_loss(student, teacher, args.top_k)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite distillation loss")
        loss.backward()
        grads = [p.grad for p in cartridge.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in grads):
            raise RuntimeError("Missing or non-finite cartridge gradients")
        torch.nn.utils.clip_grad_norm_(cartridge.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        print(json.dumps({"step": step + 1, "distillation_kl": losses[-1]}), flush=True)
        del teacher, student, loss
    meta = dict(model_id=model_id, model_revision=getattr(model.config, "_commit_hash", None),
                config_hash=fingerprint(model.config), corpus_hash=digest(text),
                cartridge_tokens=args.cartridge_tokens, steps=args.steps, lr=args.lr,
                top_k=args.top_k, seed=args.seed, losses=losses,
                objective="teacher_top_k_plus_tail_kl", position_convention="prefix_offset",
                torch_version=str(torch.__version__))
    cartridge.save(output, meta)
    print(f"Saved {output}", flush=True)

@torch.no_grad()
def evaluate(args, model, tokenizer, model_id):
    text = document(args.document)
    cartridge, meta = Cartridge.load(args.checkpoint, model, model_id)
    if meta["corpus_hash"] != digest(text):
        raise ValueError("Evaluation corpus differs from checkpoint corpus")
    rows = read_jsonl(args.data)
    if not rows:
        raise ValueError("Evaluation data is empty")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        for row in rows:
            history = [{"role": "user", "content": row["question"]}]
            result = {"question": row["question"], "reference": row["answer"]}
            target = tokenizer.encode(row["answer"], add_special_tokens=False)
            for mode in ["full_context", "no_context", "cartridge"]:
                ids = prompt_ids(tokenizer, history, text if mode == "full_context" else None)
                cache = cartridge if mode == "cartridge" else None
                logits = response_logits(model, ids, target, cache)
                nll = F.cross_entropy(logits, tensor(target, model)[0]).item()
                generated = decode(model, tokenizer, ids, args.answer_tokens, cache)
                answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
                result[mode] = {"nll": nll, "answer": answer,
                                "exact_match": answer.casefold() == row["answer"].strip().casefold()}
            f.write(json.dumps(result) + "\n")
            f.flush()
            print(json.dumps(result), flush=True)

def main():
    parser = argparse.ArgumentParser(description="Corpus-specific trainable KV cartridges")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["synthesize", "train", "evaluate", "chat"]:
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="granite, qwen2, qwen3, HF ID or local path")
        p.add_argument("--revision", default=None)
        p.add_argument("--device", default="cuda")
        p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
        p.add_argument("--offline", action="store_true")
        p.add_argument("--seed", type=int, default=42)
        if name != "chat":
            p.add_argument("--document", required=True, help="UTF-8 text corpus")
            p.add_argument("--output", required=True)
        if name in {"train", "evaluate"}:
            p.add_argument("--data", required=True)
        if name in {"evaluate", "chat"}:
            p.add_argument("--checkpoint", required=True)
        if name in {"synthesize", "evaluate", "chat"}:
            p.add_argument("--answer-tokens", type=int, default=128)
        if name == "synthesize":
            p.add_argument("--samples", type=int, default=32)
            p.add_argument("--turns", type=int, default=2)
            p.add_argument("--chunk-tokens", type=int, default=1024)
            p.add_argument("--question-tokens", type=int, default=48)
        if name == "train":
            p.add_argument("--cartridge-tokens", type=int, default=128)
            p.add_argument("--steps", type=int, default=100)
            p.add_argument("--lr", type=float, default=0.02)
            p.add_argument("--top-k", type=int, default=20)
        if name == "chat":
            p.add_argument("--question", required=True)
    args = parser.parse_args()
    if getattr(args, "chunk_tokens", 1) < 1:
        parser.error("chunk-tokens must be positive")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model, tokenizer, model_id = load_model(args.model, args.device, args.dtype, args.offline, args.revision)
    if args.command == "synthesize":
        synthesize(args, model, tokenizer, model_id)
    elif args.command == "train":
        train(args, model, tokenizer, model_id)
    elif args.command == "evaluate":
        evaluate(args, model, tokenizer, model_id)
    else:
        cartridge, _ = Cartridge.load(args.checkpoint, model, model_id)
        ids = prompt_ids(tokenizer, [{"role": "user", "content": args.question}])
        answer = decode(model, tokenizer, ids, args.answer_tokens, cartridge)
        print(tokenizer.decode(answer, skip_special_tokens=True))

if __name__ == "__main__":
    main()
