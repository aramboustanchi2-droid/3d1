import json, os
from pathlib import Path
from cad3d.super_ai.brain import SuperAIBrain

BASE = Path(__file__).parent
MERGED = BASE / 'merged_corpus.jsonl'
SUBSETS = [
    BASE/'architecture'/'processed.jsonl',
    BASE/'structure'/'processed.jsonl',
    BASE/'urban_planning'/'processed.jsonl'
]

def merge():
    with open(MERGED, 'w', encoding='utf-8') as out:
        for p in SUBSETS:
            if not p.exists():
                continue
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        out.write(line)
    return MERGED

if __name__ == '__main__':
    merged_path = merge()
    brain = SuperAIBrain()
    result = brain.train_system(str(merged_path))
    print('Training result:', result)
