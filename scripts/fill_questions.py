"""为 faux_pas / implicature 标注生成 `question`（内容锚定、不泄漏答案）。

- faux_pas：gold 为 "X said something ... inappropriate..."; 提问不引用正文，
  只问"是否有人说不该说的话/谁/为何不妥"，使模型需自行定位失礼者与原因。
- implicature：gold 为 "这句话(引文)表达的言外之意是：..."; 提取 critical_event
  台词的说话内容（去掉 "说话人：" 前缀）作为引文，问"这句话的言外之意是什么"。
  引文是题干背景而非答案，故不泄漏 gold 的"言外之意"本身。

用法： python -X utf8 scripts/fill_questions.py [--write]
不带 --write 时仅打印将生成的内容供人工审阅。
"""
import json, glob, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ANNOTATED = os.path.join("data", "annotated")
TARGET_TASKS = ("faux_pas", "implicature")

FAUX_QUESTION = (
    "In this story, did anyone say something they should not have said? "
    "If so, who said it and why was it inappropriate?"
)
IMP_FALLBACK = "这段话的言外之意（潜台词）是什么？请用一句话直接说明说话人真正想表达的意思。"

_SPEAKER_SEP = re.compile(r"^[^：:]{1,8}[:：]\s*(.+)$")


def _utterance(sentence_text: str) -> str | None:
    m = _SPEAKER_SEP.match(sentence_text.strip())
    if m and m.group(1).strip():
        return m.group(1).strip()
    return sentence_text.strip() if sentence_text.strip() else None


def make_question(annotation: dict) -> str:
    task = annotation.get("task")
    if task == "faux_pas":
        return FAUX_QUESTION
    if task != "implicature":
        raise ValueError(f"unexpected task {task!r}")
    # 找 critical_event（回退：critical_sentence 对应句）
    crit_id = annotation.get("critical_sentence")
    text = None
    for s in annotation.get("sentences", []):
        if s.get("function") == "critical_event" or s.get("id") == crit_id:
            text = s.get("text")
            break
    utt = _utterance(text) if text else None
    if utt:
        return f"这句话「{utt}」的言外之意（潜台词）是什么？请用一句话直接说明说话人真正想表达的意思。"
    return IMP_FALLBACK


def main() -> int:
    write = "--write" in sys.argv
    files = sorted(glob.glob(os.path.join(ANNOTATED, "*.json")))
    todo, skip = [], []
    for f in files:
        a = json.load(open(f, encoding="utf-8"))
        if a.get("task") not in TARGET_TASKS:
            continue
        if a.get("question"):
            skip.append(os.path.basename(f))
            continue
        todo.append((f, a))
    print(f"[fill_questions] target files={len(todo)} already-has-question={len(skip)}")
    preview = 0
    for f, a in todo:
        q = make_question(a)
        if not write:
            if preview < 8:
                print("  ", os.path.basename(f), "->", q[:150])
                preview += 1
            continue
        a["question"] = q
        with open(f, "w", encoding="utf-8") as fh:
            json.dump(a, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    if write:
        # 校验
        bad = 0
        for f, a in todo:
            chk = json.load(open(f, encoding="utf-8"))
            if not chk.get("question"):
                bad += 1
        print(f"[fill_questions] wrote question to {len(todo)} files; still-empty={bad}")
    else:
        print("[fill_questions] dry-run (no write). 用 --write 生效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
