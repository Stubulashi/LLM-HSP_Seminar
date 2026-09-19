"""docs/submission/report_final/report_final.md（中英对照）→ report_final/report_draft.tex（仅英文）同步与校验工具。

与 v2（scripts/sync_report_draft_v2.py）的关系：解析与校验规则完全一致（v2 脚本及其产物保持不动），
做以下工程化调整：
  1) 默认路径改指 docs/submission/report_final/（逐章推进的正式报告稿）；
  2) 新增可选参数 --md / --tex，便于后续章节或其它目录复用同一套同步机制；
  3) 排版增强：彩色链接与 PDF 元数据、节级 PDF 书签（\\phantomsection\\addcontentsline）、
     裸 URL → \\url、*…* → \\textit、图注 \\caption*（消除 article 自动编号与手写编号重复）；
  4) 默认输入改指正式版 report_final.md，并支持“无标记正式版”解析（2026-09-19）。

两种输入模式（自动识别：出现 [Pn-X] 标记即模式 A）：
  A. 带标记草稿：`**[Pn-CN]**` / `**[Pn-EN]**`，编号从 1 起连续唯一、中英成对（留档备份用）；
  B. 无标记正式版（默认输入）：普通文本行即段落，按“先 CN 后 EN、严格交替、偶数成对”
     校验，配对编号按出现顺序派生（仅用于 tex 的 % [Pn] 注释）。
  - 仅提取英文段落进入 tex（中文与说明性文本不进入）；
  - 节标题 `## N. English / 中文` → \\section*；小节 `### N.M English / 中文` → \\subsection*；
  - md 图片行 `![caption](path)` → figure 环境（相对路径原样写入）；
  - tex 与 md 英文逐段一致（--check 校验）。

同步规则：tex 为生成物（勿手改），每次运行脚本即与 md 英文部分完全一致。

用法：
  python -X utf8 scripts/sync_report_draft_v3.py                    # 生成/刷新 tex
  python -X utf8 scripts/sync_report_draft_v3.py --check            # 仅校验，不一致返回非 0
  python -X utf8 scripts/sync_report_draft_v3.py --md <md路径> --tex <tex路径>   # 覆盖默认路径
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

# 常用 Unicode → LaTeX 映射（先转义后映射；映射引入的 $ { } 为安全代码）
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
    """提取裸 URL → (带占位符的文本, URL 列表)；尾随标点/引号保留在文本中。"""
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
    """先保护裸 URL，再做单遍字符级转义、Unicode 映射与 *…*→\\textit，最后还原 URL。"""
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
    """解析 md → (sections, problems, marked)。

    模式 A（marked=True，带标记草稿）：**[Pn-CN]** / **[Pn-EN]**，编号从 1 起连续唯一、中英成对。
    模式 B（marked=False，无标记正式版）：普通文本行即段落，按"先 CN 后 EN、严格交替、
    偶数成对"校验，配对编号按出现顺序派生（仅用于 tex 的 % [Pn] 注释）。
    模式自动识别（全文出现任一 [Pn-X] 标记即按模式 A）。
    """
    marked = bool(PAR_RE.search(text))
    sections = []
    problems = []
    cur = None
    pending = None  # (num, lang, lines)
    seq = 0         # 模式 B 的派生编号（每对共用）

    def has_cjk(s: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", s))

    def flush_para():
        nonlocal pending
        if pending is None:
            return
        num, lang, lines = pending
        body = " ".join(seg.strip() for seg in lines if seg.strip())
        if not body:
            problems.append(f"[P{num}-{lang}] 段落内容为空")
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
                problems.append(f"[{m_sub.group(1)}] 小节出现在任何节标题之前")
                continue
            cur["items"].append({"kind": "sub", "number": m_sub.group(1),
                                 "en": m_sub.group(2).strip(), "cn": m_sub.group(3).strip()})
        elif m_img:
            flush_para()
            if cur is None:
                problems.append(f"[IMG] {m_img.group(2)} 出现在任何节标题之前")
                continue
            cur["items"].append({"kind": "fig", "alt": m_img.group(1).strip(),
                                 "path": m_img.group(2).strip()})
        elif m_par:
            flush_para()
            if cur is None:
                problems.append(f"[P{m_par.group(1)}-{m_par.group(2)}] 出现在任何节标题之前")
                cur = {"number": "?", "en": "(no section)", "cn": "", "items": []}
                sections.append(cur)
            pending = (int(m_par.group(1)), m_par.group(2), [m_par.group(3)])
        elif marked:
            # 模式 A：普通行仅作为段的续行
            if pending is not None:
                pending[2].append(line)
        else:
            # 模式 B：结构行（空行/#/>/|）刷新段落；其余文本行成段或续行
            s = line.strip()
            if not s or s.startswith("#") or s.startswith(">") or s.startswith("|"):
                flush_para()
                continue
            if pending is not None:
                pending[2].append(s)
                continue
            if cur is None:
                problems.append("段落出现在任何节标题之前")
                continue
            lang = "CN" if has_cjk(s) else "EN"
            if lang == "CN":
                seq += 1
            pending = (seq, lang, [s])
    flush_para()

    if marked:
        # 模式 A 校验：编号从 1 起连续唯一、中英成对
        nums = {}
        for sec in sections:
            for item in sec["items"]:
                if item["kind"] == "para":
                    nums.setdefault(item["num"], []).append(item["lang"])
        for num in sorted(nums):
            langs = nums[num]
            if langs.count("CN") != 1 or langs.count("EN") != 1:
                problems.append(f"[P{num}] 中英未成对：{langs}")
        if nums:
            expect = list(range(1, max(nums) + 1))
            if sorted(nums) != expect:
                missing = sorted(set(expect) - set(nums))
                problems.append(f"编号不连续，缺失：{missing}")
    else:
        # 模式 B 校验：先 CN 后 EN、严格交替、偶数成对
        for sec in sections:
            paras = [it for it in sec["items"] if it["kind"] == "para"]
            for idx, it in enumerate(paras):
                want = "CN" if idx % 2 == 0 else "EN"
                if it["lang"] != want:
                    problems.append(
                        f"节 {sec['number']} 第 {idx + 1} 段应为 {want}，实际 {it['lang']}（须中英交替）")
            if len(paras) % 2:
                problems.append(f"节 {sec['number']} 段落数为奇数（{len(paras)}），中英未成对")
    for sec in sections:
        if not any(item["kind"] == "para" for item in sec["items"]):
            problems.append(f"节 {sec['number']}. {sec['en']} 下没有任何段落")
    return sections, problems, marked


def build_tex(sections, generated_line: str) -> str:
    """由英文章节/小节/段落/图构建完整 tex。generated_line 为空时省略时间戳（供 --check 比对）。"""
    head = [
        "%% ============================================================================",
        "%% AUTOMATICALLY GENERATED — DO NOT EDIT MANUALLY / 本文件由脚本自动生成，请勿手改",
        "%% Source  : docs/submission/report_final/report_final.md  (仅提取英文段落)",
        "%% Rebuild : python -X utf8 scripts/sync_report_draft_v3.py",
    ]
    if generated_line:
        head.append(f"%% {generated_line}")
    head += [
        "%% Match   : 每段前 %% [Pn] 注释为按出现顺序派生的配对编号（无标记模式）",
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
    """读取 `--name VALUE` 形式的可选路径参数。"""
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
    print(f"[sync_v3] 解析：{len(sections)} 节 / {para_total} 段 / 英文 {en_total} 段 / 图 {fig_total}（{'标记模式' if marked else '无标记模式'}）")

    # 未映射的非 ASCII 警告（不影响生成，仅提示）
    unmapped = set()
    for sec in sections:
        for item in sec["items"]:
            if item["kind"] != "para" or item["lang"] != "EN":
                continue
            for ch in item["text"]:
                if ord(ch) > 0x24F and ch not in UNICODE_MAP:
                    unmapped.add(ch)
    if unmapped:
        print(f"[sync_v3] 警告：tex 中保留的未映射非 ASCII 字符 {sorted(unmapped)}")

    # 图片路径存在性检查（相对 md 所在目录解析）
    for sec in sections:
        for item in sec["items"]:
            if item["kind"] == "fig":
                if not (md_path.parent / item["path"]).exists():
                    problems.append(f"图片不存在：{item['path']}")

    if problems:
        print("[sync_v3] 结构错误：")
        for item in problems:
            print("  -", item)
        return 2

    generated = datetime.now().strftime("Generated: %Y-%m-%d %H:%M")
    expected_tex = build_tex(sections, generated_line="")  # 比对用（无时间戳）
    write_tex = build_tex(sections, generated_line=generated)

    if check_only:
        if not tex_path.exists():
            print(f"[sync_v3] tex 不存在：{tex_path}")
            return 1
        actual = tex_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        actual = "\n".join(
            ln for ln in actual.split("\n") if not ln.startswith("%% Generated:")
        )
        if actual.rstrip() != expected_tex.rstrip():
            print("[sync_v3] 不一致：tex 与 md 英文部分存在差异（tex 可能被手改或未同步）")
            a = actual.rstrip().split("\n")
            e = expected_tex.rstrip().split("\n")
            for i in range(max(len(a), len(e))):
                la = a[i] if i < len(a) else "<无>"
                le = e[i] if i < len(e) else "<无>"
                if la != le:
                    print(f"  首个差异（第 {i + 1} 行）:\n    tex : {la}\n    md  : {le}")
                    break
            return 1
        print("[sync_v3] 一致：tex 与 md 英文部分完全对应")
        return 0

    tex_path.write_text(write_tex, encoding="utf-8", newline="\n")
    print(f"[sync_v3] 已生成 {tex_path}（{len(write_tex.splitlines())} 行，{generated}）")
    print("[sync_v3] 提示：运行 --check 可随时校验一致性")
    return 0


if __name__ == "__main__":
    sys.exit(main())
