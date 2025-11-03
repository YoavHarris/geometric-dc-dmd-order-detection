import json
import os
from typing import Iterable, List, Dict, Any

DEFAULT_STATE: Dict[str, Any] = {
    "succeeded": [],
    "failed": [],
    "combined": [],
}


# ────────────────────────────── IO HELPERS ──────────────────────────────────
def load_state(state_path: str) -> dict:
    if not os.path.exists(state_path):
        return DEFAULT_STATE.copy()
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict, state_path: str) -> None:
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ────────────────────────────── SUCCEEDED ───────────────────────────────────
def mark_succeeded(ids: Iterable[int], state_path: str) -> None:
    st = load_state(state_path)
    st["succeeded"] = sorted(set(st.get("succeeded", [])) | set(ids))
    save_state(st, state_path)


def get_succeeded(state_path: str) -> List[int]:
    return load_state(state_path).get("succeeded", [])


# ─────────────────────────────── FAILED ─────────────────────────────────────
def mark_failed(ids: Iterable[int], state_path: str) -> None:
    st = load_state(state_path)
    st["failed"] = sorted(set(ids))
    save_state(st, state_path)


def get_failed(state_path: str) -> List[int]:
    return load_state(state_path).get("failed", [])


# ───────────────────────────── COMBINED ─────────────────────────────────────
def mark_combined(ids: Iterable[int], state_path: str) -> None:
    st = load_state(state_path)
    st["combined"] = sorted(set(st.get("combined", [])) | set(ids))
    save_state(st, state_path)


def get_combined(state_path: str) -> List[int]:
    return load_state(state_path).get("combined", [])
