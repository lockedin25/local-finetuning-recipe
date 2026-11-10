"""Clean the monthly memory cards for the system prompt (Jarvis API, DeepSeek V4.1 Flash, thinking off).

Removes exact dates, day numbers, percentiles and markdown (the summaries got dates wrong), keeps durable facts,
writes first person from Alex's side, <= 120 words. Counts only are printed.
  python3 -I src/clean_cards.py   ->  ~/LocalData/hinglish-clone/api_out/cards_clean.json
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import jarvis_reason as J  # noqa: E402

OUT = J.OUT
PROMPT = 'Rewrite the background card as a short first-person note from Alex to Alex. Sam is the other fictional participant.\nRules:\n- At most 120 words, plain sentences, no markdown, bullets, or headings.\n- Remove exact dates, day numbers, times, percentiles, scores, and prices. Prefer relative wording.\n- Keep only durable facts, current circumstances, preferences, and unresolved threads supported by the input.\n- Preserve speaker identity: I/me refers to Alex, and Sam refers to the other participant.\n- Do not introduce surnames, nicknames, third-party identities, locations, or relationship facts that the input does not supply.\n- No digits. Do not add unsupported facts or explicit descriptions.\nOutput only the note.'












DIGITS = re.compile(r"\d")


def main():
    cards = json.load(open(OUT / "cards.json"))
    llm = J.LLM(J.MODEL)
    sp = J.SamplingParams(temperature=0.2, max_tokens=400)
    months = sorted(cards)
    convs = [[{"role": "system", "content": PROMPT}, {"role": "user", "content": cards[m]["card"]}] for m in months]
    outs = llm.chat(convs, sp, use_tqdm=False)
    clean, redo = {}, []
    for m, o in zip(months, outs):
        t = o.outputs[0].text.strip()
        clean[m] = t
        if len(t.split()) > 130 or "**" in t or bool(DIGITS.search(t)) or "UNSUPPORTED_IDENTITY" in t or "I'm Sam" in t or "she's alex" in t.lower():
            redo.append(m)
    if redo:
        outs = llm.chat([[{"role": "system", "content": PROMPT},
                          {"role": "user", "content": cards[m]["card"] + "\n\n(Strictly: no numbers, no markdown, under 120 words.)"}]
                         for m in redo], sp, use_tqdm=False)
        for m, o in zip(redo, outs):
            clean[m] = o.outputs[0].text.strip()
    json.dump(clean, open(OUT / "cards_clean.json", "w"), ensure_ascii=False, indent=1)
    w = [len(v.split()) for v in clean.values()]
    print(f"cleaned {len(clean)} cards | words min {min(w)} max {max(w)} | with digits "
          f"{sum(bool(DIGITS.search(v)) for v in clean.values())} | markdown {sum('**' in v for v in clean.values())} "
          f"| redone {len(redo)} | cost ${J.cost():.3f}")


if __name__ == "__main__":
    main()
