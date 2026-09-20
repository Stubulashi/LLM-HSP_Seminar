"""docs/submission/report_final/report_final.md (bilingual CN/EN) → report_final/report_draft.tex (English only) sync and check tool.

Relation to v2 (scripts/sync_report_draft_v2.py): the parsing and validation rules are identical
(v2 and its outputs stay untouched); the engineering changes here are:
  1) default paths now point at docs/submission/report_final/ (the final report written chapter by chapter);
  2) new optional --md / --tex arguments, so later chapters or other directories can reuse the same sync mechanism;
  3) typesetting enhancements: coloured links and PDF metadata, per-section PDF bookmarks
     (\\phantomsection\\addcontentsline), bare URLs → \\url, *…* → \\textit, figure captions via
     \\caption* (removes the duplicate between the article class auto-numbering and handwritten numbers);
  4) the default input is now the final report_final.md, with support for the "untagged final version" (2026-09-19).

Two input modes (auto-detected: any [Pn-X] tag selects mode A):
  A. tagged draft: `**[Pn-CN]**` / `**[Pn-EN]**`, numbers starting at 1, unique and continuous, CN/EN paired (kept for the archive);
  B. untagged final version (the default input): plain text lines are paragraphs, validated as
     "CN first, then EN, strictly alternating, even pairs"; pair numbers are derived from the order
     of appearance (used only for the % [Pn] comments in the tex).
  - only English paragraphs enter the tex (Chinese and explanatory text do not);
  - section headings `## N. English / <Chinese title>` → \\section*; subsections `### N.M English / <Chinese title>` → \\subsection*;
  - md image lines `![caption](path)` → figure environments (relative paths written through as-is);
  - tex and md agree paragraph by paragraph in English (verified by --check).

Sync rule: the tex file is a generated artifact (never edit by hand); every run makes it match
the English part of the md exactly.

Usage:
  python -X utf8 scripts/sync_report_draft_v3.py                    # generate / refresh the tex
  python -X utf8 scripts/sync_report_draft_v3.py --check            # check only; non-zero exit on mismatch
  python -X utf8 scripts/sync_report_draft_v3.py --md <md> --tex <tex>   # override the default paths
"""
import re
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_MD = Path("docs/submission/report_final/report_final.md")
DEFAULT_TEX = Path("docs/submission/report_final/report_draft.tex")

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


URL_RE = re.compile(r'https?://[^\s<>\"]+')
_URL_TRAIL = {",", ".", ";", ":", ")", "]", "}", "`", "'", '"'}


def _protect_urls(text: str):
    """Extract bare URLs → (text with placeholders, list of URLs); trailing punctuation/quotes stay in the text."""
    urls = []

    def repl(m):
        u, tail = m.group(0), ""
        while u and u[-1] in _URL_TRAIL:
            tail = u[-1] + tail
            u = u[:-1]
        urls.append(u)
        return f"\x00URL{len(urls) - 1}\x00" + tail

    return URL_RE.sub(repl, text), urls


def latex_escape(text: str) -> str:
    """Protect bare URLs first, then do a single character-level escape pass, Unicode mapping and *…*→\\textit, restoring the URLs at the end."""
    text, urls = _protect_urls(text)
    out = []
    for ch in text:
        if ch in ESC_CHARS:
            out.append(ESC_CHARS[ch])
        else:
            out.append(ch)
    text = "".join(out)
    for src, dst in UNICODE_MAP.items():
        text = text.replace(src, dst)
    text = re.sub(r"\*([^*\n]+)\*", lambda m: r"\textit{" + m.group(1) + "}", text)
    for i, u in enumerate(urls):
        text = text.replace(f"\x00URL{i}\x00", r"\url{" + u + "}")
    return text


def parse_md(text: str):
    """Parse md → (sections, problems, marked).

    Mode A (marked=True, a tagged draft): **[Pn-CN]** / **[Pn-EN]** pairs, numbers starting at 1,
    unique and continuous, CN/EN paired.
    Mode B (marked=False, the untagged final version): plain text lines are paragraphs, validated
    as "CN first, then EN, strictly alternating, even pairs"; pair numbers are derived from the
    order of appearance (used only for the % [Pn] comments in the tex file).
    The mode is auto-detected (any [Pn-X] tag in the text selects mode A).
    """
    marked = bool(PAR_RE.search(text))
    sections = []
    problems = []
    cur = None
    pending = None  # (num, lang, lines)
    seq = 0         # derived pair number for mode B (shared by each pair)

    def has_cjk(s: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", s))

    def flush_para():
        nonlocal pending
        if pending is None:
            return
        num, lang, lines = pending
        body = " ".join(seg.strip() for seg in lines if seg.strip())
        if not body:
            problems.append(f"[P{num}-{lang}] paragraph body is empty")
        if cur is not None:
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
        elif m_img:
            flush_para()
            if cur is None:
                problems.append(f"[IMG] {m_img.group(2)} appears before any section heading")
                continue
            cur["items"].append({"kind": "fig", "alt": m_img.group(1).strip(),
                                 "path": m_img.group(2).strip()})
        elif m_par:
            flush_para()
            if cur is None:
                problems.append(f"[P{m_par.group(1)}-{m_par.group(2)}] appears before any section heading")
                cur = {"number": "?", "en": "(no section)", "cn": "", "items": []}
                sections.append(cur)
            pending = (int(m_par.group(1)), m_par.group(2), [m_par.group(3)])
        elif marked:
            # mode A: a plain line only continues the current paragraph
            if pending is not None:
                pending[2].append(line)
        else:
            # mode B: structural lines (blank/#/>/|) flush the paragraph; other text lines start or continue one
            s = line.strip()
            if not s or s.startswith("#") or s.startswith(">") or s.startswith("|"):
                flush_para()
                continue
            if pending is not None:
                pending[2].append(s)
                continue
            if cur is None:
                problems.append("a paragraph appears before any section heading")
                continue
            lang = "CN" if has_cjk(s) else "EN"
            if lang == "CN":
                seq += 1
            pending = (seq, lang, [s])
    flush_para()

    if marked:
        # mode A validation: numbers start at 1, continuous and unique, CN/EN paired
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
    else:
        # mode B validation: CN first, then EN, strictly alternating, even pairs
        for sec in sections:
            paras = [it for it in sec["items"] if it["kind"] == "para"]
            for idx, it in enumerate(paras):
                want = "CN" if idx % 2 == 0 else "EN"
                if it["lang"] != want:
                    problems.append(
                        f"section {sec['number']} paragraph {idx + 1} should be {want}, found {it['lang']} (CN/EN must alternate)")
            if len(paras) % 2:
                problems.append(f"section {sec['number']} has an odd number of paragraphs ({len(paras)}); CN/EN not paired")
    for sec in sections:
        if not any(item["kind"] == "para" for item in sec["items"]):
            problems.append(f"section {sec['number']}. {sec['en']} has no paragraphs")
    return sections, problems, marked


def build_tex(sections, generated_line: str) -> str:
    """Build the complete tex from the English sections/subsections/paragraphs/figures. An empty generated_line omits the timestamp (for --check comparison)."""
    head = [
        "%% ============================================================================",
        "%% AUTOMATICALLY GENERATED — DO NOT EDIT MANUALLY / this file is auto-generated by script; do not edit by hand",
        "%% Source  : docs/submission/report_final/report_final.md  (English paragraphs only)",
        "%% Rebuild : python -X utf8 scripts/sync_report_draft_v3.py",
    ]
    if generated_line:
        head.append(f"%% {generated_line}")
    head += [
        "%% Match   : the %% [Pn] comment before each paragraph is the pair number derived from the order of appearance (untagged mode)",
        "%% Compile : pdflatex report_draft.tex  (run twice; TeX Live / MiKTeX)",
        "%% ============================================================================",
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[a4paper,margin=2.4cm]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{microtype}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{caption}",
        r"\usepackage[colorlinks=true,urlcolor=blue,linkcolor=blue!60!black,citecolor=blue]{hyperref}",
        r"\setlength{\parskip}{0.5em}",
        r"\setlength{\parindent}{0pt}",
        r"\title{Modeling Incremental Pragmatic Inference in Open-Source Large Language Models}",
        r"\author{IPLE final report --- English typeset copy (auto-generated)}",
        r"\hypersetup{pdftitle={Modeling Incremental Pragmatic Inference in Open-Source Large Language Models}, pdfauthor={IPLE final report (auto-generated)}, pdfsubject={Incremental pragmatic evaluation of open-source LLMs}}",
        r"\date{September 2026}",
        r"\begin{document}",
        r"\maketitle",
        "",
    ]
    body = []
    for sec in sections:
        body.append("% " + "-" * 74)
        body.append(r"\phantomsection")
        body.append(rf"\addcontentsline{{toc}}{{section}}{{{sec['number']}. {latex_escape(sec['en'])}}}")
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
                body.append(rf"  \caption*{{{latex_escape(item['alt'])}}}")
                body.append(r"\end{figure}")
                body.append("")
    tail = [r"\end{document}", ""]
    return "\n".join(head + body + tail)


def _opt(name: str, default: Path, args) -> Path:
    """Read an optional path argument of the form `--name VALUE`."""
    if name in args:
        i = args.index(name)
        if i + 1 < len(args):
            return Path(args[i + 1])
    return default


def main() -> int:
    args = sys.argv[1:]
    check_only = "--check" in args
    md_path = _opt("--md", DEFAULT_MD, args)
    tex_path = _opt("--tex", DEFAULT_TEX, args)

    text = md_path.read_text(encoding="utf-8")
    sections, problems, marked = parse_md(text)

    para_total = sum(1 for s in sections for i in s["items"] if i["kind"] == "para")
    en_total = sum(1 for s in sections for i in s["items"]
                   if i["kind"] == "para" and i["lang"] == "EN")
    fig_total = sum(1 for s in sections for i in s["items"] if i["kind"] == "fig")
    print(f"[sync_v3] parsed: {len(sections)} sections / {para_total} paragraphs / {en_total} English / {fig_total} figures ({'tagged mode' if marked else 'untagged mode'})")

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
        print(f"[sync_v3] warning: unmapped non-ASCII characters kept in the tex: {sorted(unmapped)}")

    # figure-path existence check (resolved relative to the md file's directory)
    for sec in sections:
        for item in sec["items"]:
            if item["kind"] == "fig":
                if not (md_path.parent / item["path"]).exists():
                    problems.append(f"figure not found: {item['path']}")

    if problems:
        print("[sync_v3] structural errors:")
        for item in problems:
            print("  -", item)
        return 2

    generated = datetime.now().strftime("Generated: %Y-%m-%d %H:%M")
    expected_tex = build_tex(sections, generated_line="")  # for comparison (no timestamp)
    write_tex = build_tex(sections, generated_line=generated)

    if check_only:
        if not tex_path.exists():
            print(f"[sync_v3] tex does not exist: {tex_path}")
            return 1
        actual = tex_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        actual = "\n".join(
            ln for ln in actual.split("\n") if not ln.startswith("%% Generated:")
        )
        if actual.rstrip() != expected_tex.rstrip():
            print("[sync_v3] mismatch: the tex differs from the English part of the md (the tex may have been edited by hand or not synced)")
            a = actual.rstrip().split("\n")
            e = expected_tex.rstrip().split("\n")
            for i in range(max(len(a), len(e))):
                la = a[i] if i < len(a) else "<none>"
                le = e[i] if i < len(e) else "<none>"
                if la != le:
                    print(f"  first difference (line {i + 1}):\n    tex : {la}\n    md  : {le}")
                    break
            return 1
        print("[sync_v3] in sync: the tex matches the English part of the md exactly")
        return 0

    tex_path.write_text(write_tex, encoding="utf-8", newline="\n")
    print(f"[sync_v3] generated {tex_path} ({len(write_tex.splitlines())} lines, {generated})")
    print("[sync_v3] tip: run --check at any time to verify consistency")
    return 0


if __name__ == "__main__":
    sys.exit(main())
