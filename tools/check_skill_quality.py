#!/usr/bin/env python3
"""Check SKILL.md quality: frontmatter, description length, line count.

Usage:
    python tools/check_skill_quality.py
"""
import os, re, sys

SKILLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'skills')
MAX_DESC_LEN = 200
MAX_LINE_COUNT = 800

def check():
    issues = []
    total = 0
    for dirpath, _, filenames in os.walk(SKILLS_DIR):
        for f in filenames:
            if f == 'SKILL.md':
                total += 1
                filepath = os.path.join(dirpath, f)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                    content = fh.read()
                rel = os.path.relpath(filepath, os.path.dirname(SKILLS_DIR))
                if not content.startswith('---'):
                    issues.append(f'{rel}: missing frontmatter')
                else:
                    end_idx = content.find('---', 3)
                    if end_idx > 0:
                        fm = content[3:end_idx]
                        m = re.search(r'description:\s*(["\']?)(.*?)\1\s*\n', fm)
                        if m and len(m.group(2)) > MAX_DESC_LEN:
                            issues.append(f'{rel}: description too long ({len(m.group(2))} chars)')
                lines = content.split('\n')
                if len(lines) > MAX_LINE_COUNT:
                    issues.append(f'{rel}: too many lines ({len(lines)})')
    print(f'Checked {total} SKILL.md files')
    if issues:
        print(f'Found {len(issues)} issues:')
        for i in issues[:20]:
            print(f'  {i}')
        if len(issues) > 20:
            print(f'  ... and {len(issues) - 20} more')
        return 1
    print('All quality checks passed')
    return 0

if __name__ == '__main__':
    sys.exit(check())
