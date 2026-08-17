#!/usr/bin/env python3
"""Write [environment].docker_image into every task.toml of a dataset."""
import argparse, re
from pathlib import Path

SECTION = re.compile(r'\s*\[([^\]]+)\]')

def patch_text(text: str, image: str) -> str:
    out, section, done = [], None, False
    for ln in text.splitlines(keepends=True):
        m = SECTION.match(ln)
        if m:
            if section == "environment" and not done:
                out.append(f'docker_image = "{image}"\n')
                done = True
            section = m.group(1)
            out.append(ln)
            continue
        # Drop any pre-existing docker_image inside [environment]
        if section == "environment" and re.match(r'\s*docker_image\s*=', ln):
            continue
        out.append(ln)
    if section == "environment" and not done:
        out.append(f'docker_image = "{image}"\n')
        done = True
    if not done:
        out.append(f'\n[environment]\ndocker_image = "{image}"\n')
    return "".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="single image for every task")
    g.add_argument("--manifest", help="TSV of <task-name>\\t<image-path>")
    args = ap.parse_args()

    mapping = {}
    if args.manifest:
        for line in Path(args.manifest).read_text().splitlines():
            if line.strip():
                name, img = line.split("\t")
                mapping[name] = img

    n = 0
    for toml in sorted(Path(args.dataset).glob("*/task.toml")):
        image = args.image or mapping.get(toml.parent.name)
        if image is None:
            print(f"no image for {toml.parent.name}, skipped")
            continue
        new = patch_text(toml.read_text(), image)
        if new != toml.read_text():
            toml.write_text(new)
            n += 1
    print(f"patched {n}")

if __name__ == "__main__":
    main()