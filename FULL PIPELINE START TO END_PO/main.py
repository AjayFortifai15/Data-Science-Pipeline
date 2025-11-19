# main.py  (inside: FULL PIPELINE START TO END)
from __future__ import annotations
import argparse, subprocess, sys, time, shutil, os
from pathlib import Path
from typing import List

# Ensure stdout uses UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    os.environ["PYTHONIOENCODING"] = "utf-8"

def eprint(*a, **kw): print(*a, file=sys.stderr, **kw)

def copy_output_folder(project_root: Path) -> None:
    """
    Copy training_models_risks/Po_Invoice_Data to:
      1) Prediction Pipeline/Po_Invoice_Data
      2) <project_root>/Po_Invoice_Data  (root copy as you requested)
    Overwrites existing contents.
    """
    src = project_root / "training_models_risks" / "Po_Invoice_Data"
    if not src.exists():
        eprint(f"⚠️  Nothing to copy. Missing: {src}")
        return

    destinations = [
        project_root / "Prediction Pipeline" / "Po_Invoice_Data",
        project_root / "Po_Invoice_Data",  # extra copy at FULL PIPELINE START TO END
    ]

    for dst in destinations:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"📂 Copied {src} → {dst}")

def run_script(script_path: Path, verbose: bool) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-u", str(script_path)]
    cwd = str(script_path.parent)
    if verbose:
        print(f"• CWD: {cwd}")
        print(f"• CMD: {cmd}")
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)

def tail(txt: str, n: int = 120) -> str:
    lines = (txt or "").splitlines()
    return "\n".join(lines[-n:]) if lines else ""

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--copy-even-if-failed", action="store_true")
    ap.add_argument("--sleep", type=int, default=15)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    tm = project_root / "training_models_risks"

    SCRIPTS: List[Path] = [
        #tm / "po_data_prep.py",
        #tm / "invoice_data_prep.py",
        tm / "training_step_1.py",
        tm / "training_step_2.py",
        tm / "training_step_3.py",
        tm / "training_step_4.py",
    ]

    any_failed = False
    for i, script in enumerate(SCRIPTS, 1):
        print(f"\n▶ Step {i:02d} — {script.name}")
        if not script.exists():
            eprint(f"❌ Missing: {script}")
            any_failed = True
            if not args.continue_on_error:
                break
            else:
                continue

        cp = run_script(script, verbose=args.verbose)
        if cp.stdout:
            print(tail(cp.stdout))
        if cp.returncode != 0:
            any_failed = True
            eprint(f"❌ Exit {cp.returncode} — {script.name}")
            if cp.stderr:
                eprint("——— STDERR (tail) ———")
                eprint(tail(cp.stderr))
                eprint("———————————————")
            if not args.continue_on_error:
                break
        else:
            print(f"✅ Finished {script.name}")
            if i < len(SCRIPTS):
                print(f"⏸ Waiting {args.sleep}s …")
                time.sleep(args.sleep)

    if any_failed and not args.copy_even_if_failed:
        eprint("\n⛔ Pipeline stopped due to an error. Skipping copy.")
    else:
        print("\n📦 Copying outputs …")
        copy_output_folder(project_root)
        print("🎉 Done.")

if __name__ == "__main__":
    main()
