"""report_draft.md（中英对照）→ report_draft.tex（仅英文）同步与校验工具。

同步规则：仅提取 md 中 **[Pn-EN]** 段落生成 tex；[Pn-CN] 与其余内容不进入 tex。
一致性保证：tex 为生成物（勿手改），每次运行脚本即与 md 英文部分完全一致；
若 md 编号不连续、中英不配对或 tex 与 md 不一致，脚本会报错并给出差异。

用法：
  python -X utf8 scripts/sync_report_draft.py           # 校验结构并重新生成 tex
  python -X utf8 scripts/sync_report_draft.py --check   # 仅校验（不写文件），不一致返回非 0
"""
import re
import sys
from datetime import datetime
from pathlib import Path

MD_PATH = Path("docs/submission/report_draft.md")
TEX_PATH = Path("docs/submission/report_draft.tex")

SEC_RE = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*/\s*(.+?)\s*$")
PAR_RE = re.compile(r"^\*\*\[P(\d+)-(CN|EN)\]\*\*\s*(.*)$")

# 常用 Unicode → LaTeX 映射（先转义后映射；映射引入的 $ { } 为安全代码）
UNICODE_MAP = {
    "κ": r"$\kappa$", "Δ": r"$\Delta$", "δ": r"$\delta$", "λ": r"$\lambda$",
    "≤": r"$\leq$", "≥": r"$\geq$", "≈": r"$\approx$", "×": r"$\times$",
    "±": r"$\pm$", "−": r"$-$", "→": r"$\rightarrow$", "≠": r"$\neq$",
    "–": "--", "—": "---", "…": r"\dots{}",
    "‘": "`", "’": "'", "“": "``", "”": "''",
}
ESC_CHARS = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
             "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
             "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def latex_escape(text: str) -> str:
    """单遍字符级转义（避免二次转义），再应用 Unicode 映射。"""
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
    """解析 md → (sections, problems)；sections 为有序列表。"""
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
            problems.append(f"[P{num}-{lang}] 段落内容为空")
        cur["paras"].append({"num": num, "lang": lang, "text": body})
        pending = None

    for line in text.splitlines():
        m_sec = SEC_RE.match(line)
        m_par = PAR_RE.match(line)
        if m_sec:
            flush_para()
            cur = {"number": m_sec.group(1), "en": m_sec.group(2).strip(),
                   "cn": m_sec.group(3).strip(), "paras": []}
            sections.append(cur)
        elif m_par:
            flush_para()
            if cur is None:
                problems.append(f"[P{m_par.group(1)}-{m_par.group(2)}] 出现在任何节标题之前")
                cur = {"number": "?", "en": "(no section)", "cn": "", "paras": []}
                sections.append(cur)
            pending = (int(m_par.group(1)), m_par.group(2), [m_par.group(3)])
        elif pending is not None:
            pending[2].append(line)
    flush_para()

    # 结构校验：编号连续唯一、中英成对
    nums = {}
    for sec in sections:
        for p in sec["paras"]:
            nums.setdefault(p["num"], []).append(p["lang"])
    for num in sorted(nums):
        langs = nums[num]
        if langs.count("CN") != 1 or langs.count("EN") != 1:
            problems.append(f"[P{num}] 中英未成对：{langs}")
    if nums:
        expect = list(range(1, max(nums) + 1))
        if sorted(nums) != expect:
            missing = sorted(set(expect) - set(nums))
            problems.append(f"编号不连续，缺失：{missing}")
    for sec in sections:
        if not sec["paras"]:
            problems.append(f"节 {sec['number']}. {sec['en']} 下没有任何段落")
    return sections, problems


def build_tex(sections, generated_line: str) -> str:
    """由英文章节/段落构建完整 tex。generated_line 为空时省略时间戳（供 --check 比对）。"""
    head = [
        "%% ============================================================================",
        "%% AUTOMATICALLY GENERATED — DO NOT EDIT MANUALLY / 本文件由脚本自动生成，请勿手改",
        "%% Source  : docs/submission/report_draft.md  (仅提取 [Pn-EN] 英文段落)",
        "%% Rebuild : python -X utf8 scripts/sync_report_draft.py",
    ]
    if generated_line:
        head.append(f"%% {generated_line}")
    head += [
        "%% Match   : 每段前 %% [Pn] 注释即该段在 report_draft.md 中的编号",
        "%% ============================================================================",
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[a4paper,margin=2.4cm]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{microtype}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parskip}{0.5em}",
        r"\setlength{\parindent}{0pt}",
        r"\title{Final Report Writing Guide --- English Typeset Copy}",
        r"\author{auto-generated from \texttt{report\_draft.md}}",
        r"\date{September 2026}",
        r"\begin{document}",
        r"\maketitle",
        "",
    ]
    body = []
    for sec in sections:
        body.append("% " + "-" * 74)
        body.append(rf"\section*{{{sec['number']}. {latex_escape(sec['en'])}}}")
        for p in sec["paras"]:
            if p["lang"] == "EN":
                body.append(f"% [P{p['num']}]")
                body.append(latex_escape(p["text"]))
                body.append("")
    tail = [r"\end{document}", ""]
    return "\n".join(head + body + tail)


def main() -> int:
    check_only = "--check" in sys.argv[1:]
    text = MD_PATH.read_text(encoding="utf-8")
    sections, problems = parse_md(text)

    para_total = sum(len(s["paras"]) for s in sections)
    en_total = sum(1 for s in sections for p in s["paras"] if p["lang"] == "EN")
    print(f"[sync_report_draft] 解析：{len(sections)} 节 / {para_total} 段 / 英文 {en_total} 段")

    # 未映射的非 ASCII 警告（不影响生成，仅提示）
    unmapped = set()
    for sec in sections:
        for p in sec["paras"]:
            if p["lang"] != "EN":
                continue
            for ch in p["text"]:
                if ord(ch) > 0x24F and ch not in UNICODE_MAP:
                    unmapped.add(ch)
    if unmapped:
        print(f"[sync_report_draft] 警告：tex 中保留的未映射非 ASCII 字符 {sorted(unmapped)}")

    if problems:
        print("[sync_report_draft] 结构错误：")
        for item in problems:
            print("  -", item)
        return 2

    generated = datetime.now().strftime("Generated: %Y-%m-%d %H:%M")
    expected_tex = build_tex(sections, generated_line="")  # 比对用（无时间戳）
    write_tex = build_tex(sections, generated_line=generated)

    if check_only:
        if not TEX_PATH.exists():
            print(f"[sync_report_draft] tex 不存在：{TEX_PATH}")
            return 1
        actual = TEX_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
        actual = "\n".join(
            ln for ln in actual.split("\n") if not ln.startswith("%% Generated:")
        )
        if actual.rstrip() != expected_tex.rstrip():
            print("[sync_report_draft] 不一致：tex 与 md 英文部分存在差异（tex 可能被手改或未同步）")
            a = actual.rstrip().split("\n")
            e = expected_tex.rstrip().split("\n")
            for i in range(max(len(a), len(e))):
                la = a[i] if i < len(a) else "<无>"
                le = e[i] if i < len(e) else "<无>"
                if la != le:
                    print(f"  首个差异（第 {i+1} 行）:\n    tex : {la}\n    md  : {le}")
                    break
            return 1
        print("[sync_report_draft] 一致：tex 与 md 英文部分完全对应")
        return 0

    TEX_PATH.write_text(write_tex, encoding="utf-8", newline="\n")
    print(f"[sync_report_draft] 已生成 {TEX_PATH}（{len(write_tex.splitlines())} 行，{generated}）")
    print("[sync_report_draft] 提示：运行 --check 可随时校验一致性")
    return 0


if __name__ == "__main__":
    sys.exit(main())
