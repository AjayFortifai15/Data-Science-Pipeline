# main_prediction.py — Orchestrate step-1 ... step-11 with data handoffs
from __future__ import annotations
import argparse, time, os
from pathlib import Path
from importlib.machinery import SourceFileLoader
from importlib.util import spec_from_loader, module_from_spec
from typing import Any, Callable, Dict, List, Tuple, Union
import sys

# Ensure stdout uses UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Root = folder that contains this main_prediction.py
PROJECT_ROOT = Path(__file__).resolve().parent

# Folder that contains your step-*.py files
STEPS_DIR = PROJECT_ROOT / "Prediction Pipeline"

# Sleep between steps
DEFAULT_SLEEP_SECONDS = 15

# --------- PIPELINE SPEC (exact order you gave) ---------
PIPELINE: List[Dict[str, Any]] = [
    # step-1 returns TWO dataframes -> capture both
    {"file": "step-1.py",  "func": "data_load_and_cleaning_po",
     "inputs": [], "out": ["df_final", "df_final_invoice"]},

    {"file": "step-2.py",  "func": "preprocessing_feature_engineering_prediction",
     "inputs": ["df_final"], "out": "final_result_df"},

    {"file": "step-3.py",  "func": "vendor_information_and_extra_data_cleaning",
     "inputs": ["final_result_df"], "out": "final_result_df_2"},

    {"file": "step-4.py",  "func": "evidence_part_1",
     "inputs": ["final_result_df_2","df_final_invoice"], "out": "df_std"},

    {"file": "step-5.py",  "func": "evidence_part_2",
     "inputs": ["df_std"], "out": "df_std_2"},

    {"file": "step-6.py",  "func": "no_risk_over_ride",
     "inputs": ["df_std_2"], "out": "df_updated"},

    {"file": "step-7.py",  "func": "price_variance_data_cleaning",
     "inputs": ["df_updated"], "out": "df_updated_final"},

    {"file": "step-8.py",  "func": "ai_explanation_part_1",
     "inputs": ["df_updated_final"], "out": "df_refined"},

    {"file": "step-9.py",  "func": "ai_explanation_part_2",
     "inputs": ["df_refined"], "out": "fixed"},

    {"file": "step-10.py", "func": "update_after_llm_explanation",
     "inputs": ["df_updated_final","fixed"], "out": "doc_4"},

    # {"file": "step-11.py", "func": "db_update", "inputs": ["doc_4"], "out": None},
]
# --------------------------------------------------------

def load_module_from_path(py_path: Path, module_name: str):
    loader = SourceFileLoader(module_name, str(py_path))
    spec = spec_from_loader(module_name, loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {py_path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _save_outputs(store: Dict[str, Any],
                  out_spec: Union[str, List[str], None],
                  result: Any) -> List[str]:
    """
    Save a step's result into the store according to out_spec.
    - out_spec is str -> store[out_spec] = result
    - out_spec is list[str] & result is dict -> pick those keys
    - out_spec is list[str] & result is iterable -> zip by position; lengths must match
    - out_spec is None -> nothing saved
    Returns the list of keys saved.
    """
    saved: List[str] = []
    if out_spec is None:
        return saved

    if isinstance(out_spec, str):
        store[out_spec] = result
        saved.append(out_spec)
        return saved

    if isinstance(out_spec, (list, tuple)):
        names = list(out_spec)

        if isinstance(result, dict):
            for name in names:
                if name not in result:
                    raise KeyError(f"Function returned dict but missing key '{name}'")
                store[name] = result[name]
                saved.append(name)
            return saved

        # Treat as positional outputs
        if not hasattr(result, "__iter__"):
            raise TypeError("Function must return an iterable (tuple/list) when 'out' is a list of names.")
        result_list = list(result)
        if len(result_list) != len(names):
            raise ValueError(
                f"'out' expects {len(names)} values {names}, but function returned {len(result_list)}."
            )
        for k, v in zip(names, result_list):
            store[k] = v
            saved.append(k)
        return saved

    raise TypeError(f"'out' must be str, list/tuple, or None. Got: {type(out_spec).__name__}")

def run_step(step_idx: int, step: Dict[str, Any], store: Dict[str, Any]) -> Tuple[List[str], Any]:
    file_name = step["file"]
    func_name = step["func"]
    inputs_keys: List[str] = step.get("inputs", [])
    out_spec: Union[str, List[str], None] = step.get("out")

    py_path = (STEPS_DIR / file_name).resolve()
    if not py_path.exists():
        raise FileNotFoundError(f"[Step {step_idx:02d}] File not found: {py_path}")

    module_name = f"_pipeline_step_{step_idx}"
    mod = load_module_from_path(py_path, module_name)

    if not hasattr(mod, func_name):
        raise AttributeError(f"[Step {step_idx:02d}] Function '{func_name}' not found in {py_path.name}")

    func: Callable[..., Any] = getattr(mod, func_name)

    # Resolve inputs from previous steps
    args = []
    for k in inputs_keys:
        if k not in store:
            raise KeyError(f"[Step {step_idx:02d}] Input '{k}' not available. Check previous steps.")
        args.append(store[k])

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        os.environ["PYTHONIOENCODING"] = "utf-8"

    nice_inputs = ", ".join(inputs_keys) if inputs_keys else ""
    print(f"▶️  Step {step_idx:02d}: {py_path.name} :: {func_name}({nice_inputs})")
    result = func(*args)
    print(f"✅ Completed Step {step_idx:02d}")

    saved_keys = _save_outputs(store, out_spec, result)
    if saved_keys:
        print(f"🧩 Saved output as {saved_keys}")

    return saved_keys, result

def main():
    parser = argparse.ArgumentParser(description="Run step-1 .. step-11 pipeline with handoffs.")
    parser.add_argument("--from-step", type=int, default=1, help="Start from this step number (1-based).")
    parser.add_argument("--to-step",   type=int, default=len(PIPELINE), help="End at this step number (1-based).")
    parser.add_argument("--sleep",     type=int, default=DEFAULT_SLEEP_SECONDS, help="Seconds to sleep between steps.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep going even if a step fails.")
    parser.add_argument("--steps-dir", type=str, default=None,
                        help="Override steps directory (default: folder 'Prediction Pipeline' beside this script)")
    args = parser.parse_args()

    # Allow override of steps dir at runtime
    global STEPS_DIR
    if args.steps_dir:
        STEPS_DIR = Path(args.steps_dir).resolve()

    start_idx = max(1, args.from_step)
    end_idx   = min(len(PIPELINE), args.to_step)
    sleep_s   = max(0, args.sleep)

    print(f"\n=== Running pipeline steps {start_idx} → {end_idx} ===")
    print(f"Steps dir: {STEPS_DIR}\n")
    store: Dict[str, Any] = {}

    for i in range(start_idx, end_idx + 1):
        step = PIPELINE[i - 1]
        try:
            _saved, _res = run_step(i, step, store)
        except Exception as e:
            print(f"❌ Step {i:02d} FAILED: {e}")
            if not args.continue_on_error:
                print("Stopping due to error.")
                raise
        if i < end_idx and sleep_s > 0:
            print(f"⏸ Sleeping {sleep_s} seconds before next step...")
            time.sleep(sleep_s)

    print("\n🎉 Pipeline finished.")
    produced = list(store.keys())
    if produced:
        print("Outputs available in memory:", produced)

if __name__ == "__main__":
    main()

