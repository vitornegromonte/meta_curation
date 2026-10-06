#!/usr/bin/env python3
"""Localize the nara/ tree into a runnable standalone copy.

Copies .py/.sh (never data, never .git) from <repo>/nara into
<work>/code, then:
  1. drops `sys.path.append(<placeholder>)` lines (the runner sets
     per-stage PYTHONPATH instead);
  2. rewrites every placeholder data/output prefix to absolute $WORK paths;
  3. applies small functional patches (documented with NARA-DEV comments).

Usage:  nara_localize.py <repo-root> <work-dir>
Idempotent: re-running wipes <work>/code first. Upstream nara/ untouched.
"""

import os
import re
import shutil
import sys

PLACEHOLDERS = [
    # longest / most specific first
    ("your path to synthetic profiled_data", ""),
    ("your path to framework for the best frameworks ", ""),
    ("your path to save the indices ", ""),
    ("your path to check the evolution", ""),
    ("your path to check it up", ""),
    ("YOUR PATH TO APPEND HERE", ""),
    ("YOUR PATH TO DATASETS/ ", "datasets/"),
    ("YOUR PATH", ""),
    ("your path", ""),
    ("you root path", ""),
    ("/path to the synthetic data generaed after the profiling /", "/"),
    ("/path to you check the status /", "/"),
    ("/path to it /", "/"),
    ("/path /", "/"),
    ("/path", ""),
    ("/root to datasets prepared /", "/datasets/prepared"),
    ("/root to this fold /", "/"),
    ("/root to this fold ", "/"),
    ("root to this fold /", "/"),
    ("root to this fold ", "/"),
    ("root ot the fold/", "/"),
    ("root ot the fold ", "/"),
    ("root to it /", "/"),
    ("root to it ", "/"),
    ("/root to utils", ""),
    ("/root/datasets", "/datasets"),
    ("/root/model_status", "/model_status"),
    ("/root/results_evaluation", "/results_evaluation"),
    ("your root outs/", "/outs/"),
    ("/root", ""),
    ("PATH2SAVE", ""),
    ("/executing_generating.py", "@executing_generating.py"),  # @ = keep relative
]

SYS_PATH_RE = re.compile(r"^\s*sys\.path\.append\(.*\)\s*$")


def rewrite_text(text, work):
    out = []
    for line in text.splitlines():
        if SYS_PATH_RE.match(line):
            continue  # runner sets PYTHONPATH per stage
        for old, new in PLACEHOLDERS:
            if old in line:
                if new.startswith("@"):
                    line = line.replace(old, new[1:])
                else:
                    repl = work + new if new.startswith("/") else (work + "/" + new if new else work)
                    line = line.replace(old, repl)
                # collapse accidental double slashes (keep protocol-relative out: no URLs here)
                while "//" in line:
                    line = line.replace("//", "/")
        out.append(line)
    return "\n".join(out) + "\n"


def patch_get_parameters(text, work):
    # NARA-DEV: upstream hardcodes mismatched dataset/file; parametrize.
    old = """    df = read_data(path= ' YOUR PATH TO DATASETS/ datasets/prepared/fat' 
                    file_name='urinary_train.csv'
    )"""
    # (already placeholder-rewritten; match the rewritten form flexibly)
    text = re.sub(
        r"    df = read_data\(.*?\)\n",
        "    df = read_data(path=os.path.join(args.data_root, args.dataset),\n"
        "                    file_name=args.dataset + '_train.csv')\n",
        text,
        count=1,
        flags=re.DOTALL,
    )
    text = text.replace(
        "import argparse\n", "import argparse\nimport os\n", 1
    )
    text = text.replace(
        '    argparser.add_argument("--path2save", type= str, required=True)\n',
        '    argparser.add_argument("--path2save", type= str, required=True)\n'
        '    argparser.add_argument("--dataset", type=str, default="fat")\n'
        f'    argparser.add_argument("--data-root", type=str, default="{work}/datasets/prepared")\n',
        1,
    )
    text = text.replace(
        "study.optimize(obj_fun, n_trials=10)",
        'study.optimize(obj_fun, n_trials=int(os.environ.get("NARA_TRIALS", "10")))',
        1,
    )
    return text


def patch_decision(text):
    # NARA-DEV (no-op kept for documentation): the placeholder rewrite turns
    # 'PATH2SAVE/models/frameworks' into an absolute $WORK path, and the
    # runner always passes an absolute path2save, which os.path.join prefers.
    # So absolute path2save wins with no further change needed.
    return text


COMPAT_IMPORT = "from nara_compat import _read_csv_flex\n"
COMPAT_MODULE = '''"""Shared compat helpers for the localized nara tree (NARA-DEV)."""
import pandas as _pd


def _read_csv_flex(*candidates):
    """First candidate that parses to >1 column (';' then ',').

    Upstream mixes ';' and ',' across stages; a ';'-parse of a ',' file
    yields one column instead of failing, so sniff explicitly.
    """
    for c in candidates:
        try:
            df = _pd.read_csv(c, sep=";")
        except FileNotFoundError:
            continue
        if len(df.columns) > 1:
            return df
    return _pd.read_csv(candidates[-1])
'''


def patch_csv_flex(text, fallback_expr):
    # NARA-DEV: accept both profiled (;) and traditional (,) layouts.
    text = re.sub(
        r"pd\.read_csv\(([^,]+?),\s*sep=';'\)",
        r"_read_csv_flex(\1, " + fallback_expr + ")",
        text,
        count=1,
    )
    if "import pandas as pd" in text:
        text = text.replace("import pandas as pd", "import pandas as pd\n" + COMPAT_IMPORT, 1)
    return text


def main():
    repo, work = sys.argv[1], sys.argv[2]
    src = os.path.join(repo, "nara")
    code = os.path.join(work, "code")
    shutil.rmtree(code, ignore_errors=True)
    n = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "notebooks_to_prepare_data")]
        if ".ipynb_checkpoints" in dirpath:
            continue
        for fn in filenames:
            if not (fn.endswith(".py") or fn.endswith(".sh")) or fn.endswith(".ipynb"):
                continue
            if ".DS_Store" in fn:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, src)
            dst = os.path.join(code, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(full) as f:
                text = f.read()
            text = rewrite_text(text, work)
            if rel.endswith("get_parameters.py"):
                text = patch_get_parameters(text, work)
            if "def decision(" in text:
                text = patch_decision(text)
            if rel.endswith("after_profile_synth_data.py"):
                text = patch_csv_flex(
                    text, "f'" + work + "/generated/{args.dataset_name}/{args.model_name}.csv'")
            if rel.endswith("final_data.py"):
                text = patch_csv_flex(
                    text, "f'" + work + "/generated/{args.dataset_name}/{args.model_name}.csv'")
            with open(dst, "w") as f:
                f.write(text)
            n += 1
    with open(os.path.join(code, "nara_compat.py"), "w") as f:
        f.write(COMPAT_MODULE)
    print(f"localized {n} files -> {code}")


if __name__ == "__main__":
    main()
