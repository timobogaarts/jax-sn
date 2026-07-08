#!/usr/bin/env python3
"""Replace ad-hoc LaTeX macro names with their real-package equivalents.

Usage:
    python latex_macro_fix.py 'the equation $\vb{v} \bomega \diff x$'
    echo '$\vb{v}$' | python latex_macro_fix.py
    python latex_macro_fix.py --copy 'the equation $\vb{v} \bomega \diff x$'
"""

import re
import shutil
import subprocess
import sys

# command name (no leading backslash) -> full replacement text
REPLACEMENTS = {
    "vb": r"\mathbf",
    "bomega": r"\Omega",
    "diff": "d",
}


def fix_latex(text: str) -> str:
    for old, new in REPLACEMENTS.items():
        # match "\old" only when not followed by more letters (so \vbar isn't touched)
        text = re.sub(rf"\\{old}(?![a-zA-Z])", lambda m, new=new: new, text)
    return text


# first command found wins; each is (command, args, needs_binary_stdin)
CLIPBOARD_COMMANDS = [
    ("clip.exe", []),  # WSL -> Windows clipboard
    ("pbcopy", []),  # macOS
    ("wl-copy", []),  # Wayland
    ("xclip", ["-selection", "clipboard"]),
    ("xsel", ["--clipboard", "--input"]),
]


def copy_to_clipboard(text: str) -> bool:
    for cmd, args in CLIPBOARD_COMMANDS:
        exe = shutil.which(cmd)
        if exe:
            subprocess.run([exe, *args], input=text.encode(), check=True)
            return True
    return False


def main() -> None:
    args = sys.argv[1:]
    copy = "--copy" in args
    args = [a for a in args if a != "--copy"]

    if args:
        text = " ".join(args)
    else:
        text = sys.stdin.read()

    result = fix_latex(text)

    if copy:
        if not copy_to_clipboard(result):
            sys.exit("no clipboard utility found (tried clip.exe/pbcopy/wl-copy/xclip/xsel)")
    else:
        sys.stdout.write(result)
        if args:
            sys.stdout.write("\n")


if __name__ == "__main__":
    main()
