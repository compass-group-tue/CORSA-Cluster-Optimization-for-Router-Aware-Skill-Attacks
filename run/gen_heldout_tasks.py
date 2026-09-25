#!/usr/bin/env python3
"""Generate held-out tasks from public benchmark tasks and the pinned split.

Paraphrases retain the source task ID for environment resolution; generation
does not provision an environment. Synthetic tasks use cluster task examples
and require an explicit environment mapping for real evaluation.
MODEL selects the generator (default gpt-5.4). Output files are written to
results_heldout_<date>/<cluster>/{paraphrase,synthetic}.jsonl.
"""
import json, os, sys, re
from pathlib import Path
from datetime import datetime

from src.infra.paths import PROJECT_ROOT as P
from src.benchmarks.skillrouter_bench import load_core_tasks
from src.infra.llm_client import make_openai_client, chat_with_backoff

MODEL = os.environ.get("MODEL", "gpt-5.4")
N = int(os.environ.get("N_PER_CLUSTER", "9"))
OUT = P/f"results_heldout_{datetime.now():%Y%m%d}"
CLUSTERS = os.environ.get("CLUSTERS","").split(",") if os.environ.get("CLUSTERS") else None

def chat(oa, system, user):
    r = chat_with_backoff(oa, model=MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        response_format={"type":"json_object"})
    return json.loads(r.choices[0].message.content)

PARA_SYS = ("You reword coding-agent task instructions. Keep the EXACT same task and "
            "requirements — only change the wording/phrasing so it reads differently. "
            "Do not add or drop requirements. Output strict JSON.")
SYN_SYS  = ("You write NEW, realistic coding-agent benchmark tasks for a given domain. "
            "Each must be a genuinely DIFFERENT task (not a reword of the examples), but "
            "in the same domain and the same realistic style/length. Output strict JSON.")

def gen_paraphrases(oa, tasks, k):
    out=[]; i=0
    while len(out) < k and tasks:
        t = tasks[i % len(tasks)]; i+=1
        try:
            d = chat(oa, PARA_SYS,
                f"Reword this task instruction (same task, different words):\n\n{t['instruction_text'][:1500]}\n\n"
                'Return JSON: {"instruction": "..."}')
            instr = (d.get("instruction") or "").strip()
            if instr:
                out.append({"task_id": f"para__{t['task_id']}__{len([o for o in out if o['src']==t['task_id']])}",
                            "src": t["task_id"], "instruction_text": instr})
        except Exception as e:
            print(f"  para fail {t['task_id']}: {str(e)[:80]}")
        if i > k*4: break
    return out[:k]

def gen_synthetic(oa, cluster, tasks, k):
    examples = "\n".join(f"- {str(t['instruction_text'])[:220]}" for t in tasks[:12])
    d = chat(oa, SYN_SYS,
        f"Domain cluster: {cluster}\n\nExample real tasks from this domain:\n{examples}\n\n"
        f"Write {k} NEW realistic tasks in THIS domain, each distinct from the examples and "
        f'from each other. Return JSON: {{"tasks": ["instruction 1", "instruction 2", ...]}}')
    out=[]
    for j, instr in enumerate(d.get("tasks", [])[:k]):
        instr=(instr or "").strip()
        if instr: out.append({"task_id": f"syn__{cluster}__{j}", "src": None, "instruction_text": instr})
    return out

def main():
    oa = make_openai_client()
    all_tasks,_ = load_core_tasks()
    cats = json.loads((P/"data/task_categorization/task_categories.json").read_text())
    by={}
    for t in all_tasks:
        cl=cats.get(t["task_id"])
        if cl: by.setdefault(cl,[]).append(t)
    clusters = CLUSTERS or sorted(by)
    for cl in clusters:
        tasks = by.get(cl, [])
        if not tasks: print(f"{cl}: no tasks, skip"); continue
        d = OUT/cl; d.mkdir(parents=True, exist_ok=True)
        para = gen_paraphrases(oa, tasks, N)
        syn  = gen_synthetic(oa, cl, tasks, N)
        (d/"paraphrase.jsonl").write_text("\n".join(json.dumps(x) for x in para))
        (d/"synthetic.jsonl").write_text("\n".join(json.dumps(x) for x in syn))
        print(f"{cl}: {len(para)} paraphrase, {len(syn)} synthetic  (real tasks={len(tasks)})")

if __name__ == "__main__":
    main()
