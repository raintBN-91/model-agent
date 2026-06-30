#!/usr/bin/env python3
"""Check SKILL.md files for broken local file references.

Usage:
    python tools/check_skill_refs.py

Exits with non-zero status if any broken references are found.
"""
import os
import re
import sys

SKILLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'skills')


def check_skill_refs():
    """Check all SKILL.md files for broken local file references."""
    broken = []
    total = 0
    for dirpath, _, filenames in os.walk(SKILLS_DIR):
        for f in filenames:
            if f == 'SKILL.md':
                total += 1
                filepath = os.path.join(dirpath, f)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                    content = fh.read()
                refs = re.findall(
                    r'\((scripts/[^)]+|templates/[^)]+|references/[^)]+|assets/[^)]+)\)',
                    content,
                )
                for ref in set(refs):
                    ref_file = ref.split('#')[0]
                    ref_path = os.path.normpath(os.path.join(dirpath, ref_file))
                    if not os.path.exists(ref_path):
                        rel = os.path.relpath(filepath, os.path.dirname(SKILLS_DIR))
                        broken.append((rel, ref))

    print(f"Checked {total} SKILL.md files")
    if broken:
        print(f"Found {len(broken)} broken references:")
        for skill, ref in broken:
            print(f"  {skill} -> {ref}")
        return 1
    print("All references OK")
    return 0


if __name__ == '__main__':
    sys.exit(check_skill_refs())
