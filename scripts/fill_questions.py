"""Generate `question` for faux_pas / implicature annotations (content-anchored, without leaking the answer).

- faux_pas: the gold is "X said something ... inappropriate..."; the question does not quote the story
  text and only asks "did anyone say something they should not have said / who / why was it inappropriate",
  so the model must locate the offending speaker and the reason itself.
- implicature: the gold is "what this line (quote) implies is: ..."; the spoken content of the
  critical_event line (with any speaker-prefix removed) is extracted as the quote, and the question
  asks what the line implies. The quote is question background, not the answer, so the implied meaning
  in the gold is not leaked. The generated question text itself stays in Chinese to match the
  implicature task's data language.

Usage: python -X utf8 scripts/fill_questions.py [--write]
Without --write it only prints what would be generated, for human review.
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
    # find the critical_event (fallback: the sentence referenced by critical_sentence)
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
        # verification
        bad = 0
        for f, a in todo:
            chk = json.load(open(f, encoding="utf-8"))
            if not chk.get("question"):
                bad += 1
        print(f"[fill_questions] wrote question to {len(todo)} files; still-empty={bad}")
    else:
        print("[fill_questions] dry-run (no write). Pass --write to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
