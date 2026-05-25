#!/usr/bin/env python3

import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Optional


LOSS_RE = re.compile(r"'loss':\s*([0-9.eE+-]+)")
EPOCH_RE = re.compile(r"'epoch':\s*([0-9.eE+-]+)")


def parse_log(log_path: str) -> Dict[str, str]:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    lines = [line.rstrip() for line in content.splitlines()]
    non_empty_lines = [line for line in lines if line.strip()]

    last_loss: Optional[str] = None
    last_epoch: Optional[str] = None
    for match in LOSS_RE.finditer(content):
        last_loss = match.group(1)
    for match in EPOCH_RE.finditer(content):
        last_epoch = match.group(1)

    lower_content = content.lower()
    if "traceback (most recent call last):" in lower_content or " exits with return code = " in lower_content:
        status = "error"
    elif "finished." in lower_content or "exits successfully." in lower_content:
        status = "completed"
    elif non_empty_lines:
        status = "running_or_incomplete"
    else:
        status = "empty"

    return {
        "log_name": os.path.basename(log_path),
        "status": status,
        "last_epoch": last_epoch or "",
        "last_loss": last_loss or "",
        "last_line": non_empty_lines[-1] if non_empty_lines else "",
    }


def write_csv(rows: List[Dict[str, str]], csv_path: str) -> None:
    fieldnames = ["log_name", "status", "last_epoch", "last_loss", "last_line"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: List[Dict[str, str]], md_path: str, log_root: str) -> None:
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Ablation Log Summary\n\n")
        f.write(f"Log root: `{log_root}`\n\n")
        if not rows:
            f.write("No ablation log files were found.\n")
            return

        f.write("| Log | Status | Last Epoch | Last Loss |\n")
        f.write("| --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                f"| `{row['log_name']}` | {row['status']} | "
                f"{row['last_epoch'] or '-'} | {row['last_loss'] or '-'} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ablation run logs.")
    parser.add_argument("--log-root", required=True, help="Directory containing ablation log files.")
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix for summary outputs. Writes <prefix>.csv and <prefix>.md.",
    )
    args = parser.parse_args()

    log_paths = sorted(
        path
        for path in glob.glob(os.path.join(args.log_root, "*.log"))
        if os.path.isfile(path)
    )
    rows = [parse_log(path) for path in log_paths]

    csv_path = f"{args.output_prefix}.csv"
    md_path = f"{args.output_prefix}.md"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_csv(rows, csv_path)
    write_md(rows, md_path, args.log_root)

    print(f"[summarize_ablation_logs] wrote {csv_path}")
    print(f"[summarize_ablation_logs] wrote {md_path}")


if __name__ == "__main__":
    main()
