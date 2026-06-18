from pathlib import Path


BANNED_PATTERNS = [
    "pyautogui",
    "pynput",
    "mouse_event",
    "keybd_event",
    "SendInput",
    "ctypes.windll.user32",
    "ReadProcessMemory",
    "WriteProcessMemory",
    "OpenProcess",
    "CreateRemoteThread",
    "VirtualAllocEx",
    "inject",
    "hook",
]

IGNORED_DIRS = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    "dataset",
    "captures",
}


def should_ignore(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    return any(part in IGNORED_DIRS for part in rel.parts)


def main():
    root = Path(__file__).resolve().parent
    bad_hits = []

    for path in root.rglob("*.py"):
        if should_ignore(path, root):
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in BANNED_PATTERNS:
            if pattern.lower() in text.lower():
                bad_hits.append((path.relative_to(root), pattern))

    if bad_hits:
        print("Potentially unsafe patterns found in project code:")
        for path, pattern in bad_hits:
            print(f"- {path}: {pattern}")
        raise SystemExit(1)

    print("OK: project code has no obvious keyboard/mouse/process-memory/injection patterns.")


if __name__ == "__main__":
    main()
