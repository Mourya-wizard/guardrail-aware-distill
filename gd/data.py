import json, os
from .domains.support import SupportDomain


def get_domain(name="support"):
    return {"support": SupportDomain}[name]()


def build(domain, out="data", seed=0):
    os.makedirs(out, exist_ok=True)
    d = domain.generate_data(seed)
    for s, rows in d.items():
        with open(f"{out}/{s}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    return d


def load(split, out="data"):
    return [json.loads(l) for l in open(f"{out}/{split}.jsonl")]


if __name__ == "__main__":
    from collections import Counter
    d = build(get_domain())
    for s, rows in d.items():
        print(s, len(rows), dict(Counter(r["cls"] for r in rows)), "templates:", len({r["template_id"] for r in rows}))
