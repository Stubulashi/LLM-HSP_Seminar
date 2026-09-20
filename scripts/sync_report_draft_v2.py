"""report_draft_v2.md (bilingual CN/EN) → report_draft_v2.tex (English only) sync and check tool.

Differences from v1 (scripts/sync_report_draft.py):
  1) paths moved to docs/submission/report_draft_v2/ (the v1 files stay untouched);
  2) supports `### N.M English / <Chinese title>` subsection headings (tex output \\subsection*);
  3) supports md image lines `![caption](figs/xxx.png)`: the tex output uses figure environments (with caption and relative path);
  4) all other rules match v1: only **[Pn-EN]** paragraphs enter the tex; numbers are continuous and
     unique, CN/EN are paired, and the tex agrees paragraph by paragraph with the md English part;
     otherwise an error is raised with the differences.

Sync rule: the tex file is a generated artifact (never edit by hand); every run makes it match
the English part of the md exactly.

Usage:
  python -X utf8 scripts/sync_report_draft_v2.py           # validate structure and regenerate the tex
  python -X utf8 scripts/sync_report_draft_v2.py --check   # check only (no writes); non-zero exit on mismatch
"""
import re
import sys
from datetime import datetime
from pathlib import Path

MD_PATH = Path("docs/submission/report_draft_v2/report_draft.md")
TEX_PATH = Path("docs/submission/report_draft_v2/report_draft.tex")

SEC_RE = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*/\s*(.+?)\s*$")
SUB_RE = re.compile(r"^###\s+([\d.]+)\s+(.+?)\s*/\s*(.+?)\s*$")
PAR_RE = re.compile(r"^\*\*\[P(\d+)-(CN|EN)\]\*\*\s*(.*)$")
IMG_RE = re.compile(r"^!\[(.*)\]\((.+)\)$")

# common Unicode → LaTeX mapping (escape first, then map; the $ { } introduced by the mapping are safe code)
UNICODE_MAP = {
    "κ": r"$\kappa$", "Δ": r"$\Delta$", "δ": r"$\delta$", "λ": r"$\lambda$",
    "≤": r"$\leq$", "≥": r"$\geq$", "≈": r"$\approx$", "×": r"$\times$",
    "±": r"$\pm$", "−": r"$-$", "→": r"$\rightarrow$", "≠": r"$\neq$",
    "–": "--", "—": "---", "…": r"\dots{}",
    "‘": "`", "’": "'", "“": "``", "”": "''",
}
ESC_CHARS = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
             "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
             "^": r"\textasciicircum{}"}


def latex_escape(text: str) -> str:
    """Single character-level escape pass (avoids double escaping), then the Unicode mapping."""
    out = []
    for ch in text:
        if ch in ESC_CHARS:
            out.append(ESC_CHARS[ch])
        else:
            out.append(ch)
    text = "".join(out)
    for src, dst in UNICODE_MAP.items():
        text = text.replace(src, dst)
    return text


def parse_md(text: str):
    """Parse md → (sections, problems); sections is an ordered list and items keep their order of appearance."""
    sections = []
    problems = []
    cur = None
    pending = None  # (num, lang, lines)

    def flush_para():
        nonlocal pending
        if pending is None:
            return
        num, lang, lines = pending
        body = " ".join(seg.strip() for seg in lines if seg.strip())
        if not body:
            problems.append(f"[P{num}-{lang}] paragraph body is empty")
        cur["items"].append({"kind": "para", "num": num, "lang": lang, "text": body})
        pending = None

    for line in text.splitlines():
        m_sec = SEC_RE.match(line)
        m_sub = SUB_RE.match(line)
        m_par = PAR_RE.match(line)
        m_img = IMG_RE.match(line)
        if m_sec:
            flush_para()
            cur = {"number": m_sec.group(1), "en": m_sec.group(2).strip(),
                   "cn": m_sec.group(3).strip(), "items": []}
            sections.append(cur)
        elif m_sub:
            flush_para()
            if cur is None:
                problems.append(f"[{m_sub.group(1)}] subsection appears before any section heading")
                continue
            cur["items"].append({"kind": "sub", "number": m_sub.group(1),
                                 "en": m_sub.group(2).strip(), "cn": m_sub.group(3).strip()})
        elif m_par:
            flush_para()
            if cur is None:
                problems.append(f"[P{m_par.group(1)}-{m_par.group(2)}] appears before any section heading")
                cur = {"number": "?", "en": "(no section)", "cn": "", "items": []}
                sections.append(cur)
            pending = (int(m_par.group(1)), m_par.group(2), [m_par.group(3)])
        elif m_img:
            flush_para()
            if cur is None:
                problems.append(f"[IMG] {m_img.group(2)} appears before any section heading")
                continue
            cur["items"].append({"kind": "fig", "alt": m_img.group(1).strip(),
                                 "path": m_img.group(2).strip()})
        elif pending is not None:
            pending[2].append(line)
    flush_para()

    # structural validation: numbers continuous and unique, CN/EN paired
    nums = {}
    for sec in sections:
        for item in sec["items"]:
            if item["kind"] == "para":
                nums.setdefault(item["num"], []).append(item["lang"])
    for num in sorted(nums):
        langs = nums[num]
        if langs.count("CN") != 1 or langs.count("EN") != 1:
            problems.append(f"[P{num}] CN/EN not paired: {langs}")
    if nums:
        expect = list(range(1, max(nums) + 1))
        if sorted(nums) != expect:
            missing = sorted(set(expect) - set(nums))
            problems.append(f"numbering is not continuous; missing: {missing}")
    for sec in sections:
        if not any(item["kind"] == "para" for item in sec["items"]):
            problems.append(f"section {sec['number']}. {sec['en']} has no paragraphs")
    return sections, problems


def build_tex(sections, generated_line: str) -> str:
    """Build the complete tex from the English sections/subsections/paragraphs/figures. An empty generated_line omits the timestamp (for --check comparison)."""
    head = [
        "%% ============================================================================",
        "%% AUTOMATICALLY GENERATED — DO NOT EDIT MANUALLY / this file is auto-generated by script; do not edit by hand",
        "%% Source  : docs/submission/report_draft_v2/report_draft.md  ([Pn-EN] English paragraphs only)",
        "%% Rebuild : python -X utf8 scripts/sync_report_draft_v2.py",
    ]
    if generated_line:
        head.append(f"%% {generated_line}")
    head += [
        "%% Match   : the %% [Pn] comment before each paragraph is its number in report_draft_v2.md",
        "%% Compile : pdflatex report_draft.tex  (run twice; TeX Live / MiKTeX)",
        "%% ============================================================================",
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[a4paper,margin=2.4cm]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{microtype}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parskip}{0.5em}",
        r"\setlength{\parindent}{0pt}",
        r"\title{IPLE Final Report Draft (English Typeset Copy)}",
        r"\author{auto-generated from \texttt{report\_draft\_v2.md}}",
        r"\date{September 2026}",
        r"\begin{document}",
        r"\maketitle",
        "",
    ]
    body = []
    for sec in sections:
        body.append("% " + "-" * 74)
        body.append(rf"\section*{{{sec['number']}. {latex_escape(sec['en'])}}}")
        for item in sec["items"]:
            if item["kind"] == "sub":
                body.append(rf"\subsection*{{{latex_escape(item['en'])}}}")
            elif item["kind"] == "para" and item["lang"] == "EN":
                body.append(f"% [P{item['num']}]")
                body.append(latex_escape(item["text"]))
                body.append("")
            elif item["kind"] == "fig":
                body.append(r"\begin{figure}[t]")
                body.append(r"  \centering")
                body.append(rf"  \includegraphics[width=\linewidth]{{{item['path']}}}")
                body.append(rf"  \caption{{{latex_escape(item['alt'])}}}")
                body.append(r"\end{figure}")
                body.append("")
    tail = [r"\end{document}", ""]
    return "\n".join(head + body + tail)


def main() -> int:
    check_only = "--check" in sys.argv[1:]
    text = MD_PATH.read_text(encoding="utf-8")
    sections, problems = parse_md(text)

    para_total = sum(1 for s in sections for i in s["items"] if i["kind"] == "para")
    en_total = sum(1 for s in sections for i in s["items"]
                   if i["kind"] == "para" and i["lang"] == "EN")
    fig_total = sum(1 for s in sections for i in s["items"] if i["kind"] == "fig")
    print(f"[sync_v2] parsed: {len(sections)} sections / {para_total} paragraphs / {en_total} English / fig {fig_total}")

    # warning for unmapped non-ASCII characters (advisory only; does not affect generation)
    unmapped = set()
    for sec in sections:
        for item in sec["items"]:
            if item["kind"] != "para" or item["lang"] != "EN":
                continue
            for ch in item["text"]:
                if ord(ch) > 0x24F and ch not in UNICODE_MAP:
                    unmapped.add(ch)
    if unmapped:
        print(f"[sync_v2] warning: unmapped non-ASCII characters kept in the tex: {sorted(unmapped)}")

    # figure-path existence check (resolved relative to this directory)
    for sec in sections:
        for item in sec["items"]:
            if item["kind"] == "fig":
                if not (MD_PATH.parent / item["path"]).exists():
                    problems.append(f"figure not found: {item['path']}")

    if problems:
        print("[sync_v2] structural errors:")
        for item in problems:
            print("  -", item)
        return 2

    generated = datetime.now().strftime("Generated: %Y-%m-%d %H:%M")
    expected_tex = build_tex(sections, generated_line="")  # for comparison (no timestamp)
    write_tex = build_tex(sections, generated_line=generated)

    if check_only:
        if not TEX_PATH.exists():
            print(f"[sync_v2] tex does not exist: {TEX_PATH}")
            return 1
        actual = TEX_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
        actual = "\n".join(
            ln for ln in actual.split("\n") if not ln.startswith("%% Generated:")
        )
        if actual.rstrip() != expected_tex.rstrip():
            print("[sync_v2] mismatch: the tex differs from the English part of the md (the tex may have been edited by hand or not synced)")
            a = actual.rstrip().split("\n")
            e = expected_tex.rstrip().split("\n")
            for i in range(max(len(a), len(e))):
                la = a[i] if i < len(a) else "<none>"
                le = e[i] if i < len(e) else "<none>"
                if la != le:
                    print(f"  first difference (line {i + 1}):\n    tex : {la}\n    md  : {le}")
                    break
            return 1
        print("[sync_v2] in sync: the tex matches the English part of the md exactly")
        return 0

    TEX_PATH.write_text(write_tex, encoding="utf-8", newline="\n")
    print(f"[sync_v2] generated {TEX_PATH} ({len(write_tex.splitlines())} lines, {generated})")
    print("[sync_v2] tip: run --check at any time to verify consistency")
    return 0


if __name__ == "__main__":
    sys.exit(main())
