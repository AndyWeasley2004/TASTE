"""Rewrite retrieved imagery with a local Qwen server, then verify and merge.

rewrite retries invalid captions and repairs cached failures using the same
caption_violations predicate as verify. merge replaces the labels' imagery
column; --backup-suffix preserves the previous labels.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from openai import AsyncOpenAI

from taste import paths

BASE = paths.data("aria_midi", "deduped", "large_public_labels")
LABELS = BASE / "labels.parquet"
OUT_DIR = BASE / "imagery_rewrite"
REWRITES = OUT_DIR / "rewrites.jsonl"
MODEL = "Qwen/Qwen3.5-27B-FP8"
BASE_URL = "http://localhost:8100/v1"
CONCURRENCY = 64
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOKENS = 120
FEWSHOT_PER_PIECE = 3
MAX_ATTEMPTS = 8
MIN_WORDS = 8

# Musical vocabulary the caption must never contain (rule 4 of the rewrite
# system prompt): captions describe what the music evokes, never the music.
MUSIC = re.compile(
    r"\b(chords?|arpeggios?|arpeggiated|melod(?:y|ies|ic)|harmon(?:y|ies|ic)|"
    r"tempo|staccato|legato|pedal|ostinato|cadence|crescendo|diminuendo|"
    r"counterpoint|contrapuntal|pianists?|pianos?|keyboards?|ivory keys|"
    r"bass ?lines?|treble|chromatic|octaves?|glissando|trills?|notes?|"
    r"rhythm\w*|beats?|songs?|music\w*|tunes?|waltz\w*|symphon\w*|"
    r"sonatas?|refrains?|lullab\w*|serenades?|hymns?|jazz)\b", re.I)
CJK = re.compile(r"[一-鿿぀-ヿ가-힯Ѐ-ӿ]")

MUSIC_VIOLATION = "musical vocabulary"
CJK_VIOLATION = "non-English characters"
SHORT_VIOLATION = "too short"

SYSTEM = """\
You rewrite one-line mood captions for solo piano pieces. Each request gives
the piece's control attributes and its current caption (which may be shared
verbatim with many other pieces, so it must be rewritten freshly).

Rules:
1. Preserve the mood, emotional tone, and energy implied by the current
   caption and the attributes. Never contradict a given attribute, and never
   introduce feelings with no basis in the input. The mood must be
   unmistakable: a calm piece must never read as a triumphant celebration;
   a sad piece must never turn hopeful; an anxious piece is never serene.
2. You MAY freely invent a different concrete scene, setting, or imagery, as
   long as it evokes the same feeling. Prefer a genuinely different scene
   over paraphrasing the given one.
3. Do not name composers, specific works, films, or characters unless the
   current caption already does.
4. The caption is pure scenery: what the music evokes, never a description
   of the music. Do not mention the piano, a performer, playing, or any
   instrument, and use no musical vocabulary at all — no notes, chords,
   arpeggios, melody, harmony, tempo, and not even "rhythm", "beat", song
   or music words. Convey the attributes only through scene, motion,
   weather, objects, and feeling.
5. One or two sentences, at most 40 words, evocative and concrete. Vary your
   openings — do not begin every caption the same way.
6. Avoid stock imagery: light through windows, dust motes, spilling
   moonlight/sunlight, steaming mugs, herons in reeds, brass gears and
   clockwork, drifting autumn leaves, and rain on glass are heavily
   overused — reach for fresher scenes and senses (sound, touch, motion,
   weather, place). Captions that share a mood must not read as
   paraphrases of one another: choose a genuinely different protagonist,
   object, and place each time, as if written by a different author.
7. Each request offers a scene-domain suggestion and two concrete elements.
   Weave in what fits the mood; ignore anything that fights it — mood
   always wins. The elements are neutral: let the FULL attribute set decide
   what they do and how they move. Autumn leaves may settle one by one on
   still ground (calm or sad, slow), swirl in a bright chase (happy, fast),
   or be ripped into a violent spiral (angry). The same street, animal, or
   object should act, sound, and feel completely different under different
   attributes.
   Within the attributes, emotion outranks tempo: tempo is motion speed,
   not mood. A calm fast piece is fluent and effortless, not urgent or
   excited; a happy slow piece is contented, not mournful.
8. Output ONLY the caption text: no quotes, labels, or commentary."""

DOMAINS = [
    "city streets or transit", "open countryside or fields",
    "sea, coast, or harbor", "mountains or forest",
    "rivers, lakes, or rain", "night sky or weather",
    "home interiors", "cafés, shops, or markets", "workshops or studios",
    "libraries, schools, or archives", "gardens or parks",
    "festivals or celebrations", "travel or departures",
    "childhood or memory", "letters, photographs, or keepsakes",
    "dance or movement", "games or play", "work or craft",
    "seasons changing", "animals or birds", "machines or clockwork",
    "theater, circus, or fairground", "old towns or ruins",
    "kitchens or shared meals",
]

# (attributes-line, original caption, rewritten caption) — originals sampled
# from the retrieved captions; rewrites hand-authored to demonstrate scene reinvention
# under a preserved mood.
FEWSHOT = [
    ("emotion: sad | genre: classical | accompaniment: arpeggiated | "
     "performance: flexible connected | tempo: moderate",
     "Solitary rain droplets slide down a cool windowpane, evoking quiet "
     "solitude. A gentle stream swells with persistent hope, scattering "
     "sparkling dew before slowing into a deep basin, dissolving into the "
     "peaceful silence of twilight.",
     "An empty train platform at dusk, one figure watching the last "
     "carriage shrink into grey distance, sorrow rising and thinning like "
     "breath in cold air before settling into stillness."),
    ("emotion: calm | genre: classical | accompaniment: chordal | "
     "tempo: slow",
     "A solitary figure watches mist rise from a quiet lake at dawn, where "
     "memory washes over them with gentle longing. As darkness softens into "
     "pale light, a peaceful acceptance settles, bringing enduring patience "
     "to the still morning.",
     "Snow gathers without sound on a monastery roof while a single lamp "
     "burns in the scriptorium, time passing like pages turned by a "
     "patient, forgiving hand."),
    ("emotion: angry | genre: pop | accompaniment: ostinato | "
     "performance: steady connected | tempo: moderate",
     "A crowded boxing gym during sparring, sweat dripping on the canvas "
     "while trainers shout and heavy bags swing hard against the concrete "
     "walls in the heat.",
     "Storming down a midnight highway with headlights slicing the rain, "
     "jaw set, the same furious thought returning with every mile marker "
     "on the rain-black asphalt."),
    ("emotion: happy | genre: classical | accompaniment: chordal | "
     "performance: flexible connected | tempo: fast",
     "A quiet evening in a dimly lit ballroom, where the soft glow of "
     "candlelight reflects off polished floors as the gentle rhythm of the "
     "waltz carries memories of a bygone era.",
     "Confetti still drifting in the square after the parade has passed, "
     "children chasing ribbons between café tables while the whole square "
     "sways up onto its toes."),
    ("emotion: anxious | genre: classical | accompaniment: contrapuntal | "
     "performance: flexible connected | tempo: moderate",
     "A misty morning on a desolate coast shifts from calm ripples to a "
     "darkening horizon, then erupts into a violent storm of crashing waves "
     "and heavy rain before settling into a relentless, drenched aftermath.",
     "Two arguments chase each other through a locked house, voices "
     "overlapping in narrowing corridors, doors slamming closer and closer "
     "until the walls themselves seem to lean in and listen."),
    ("genre: pop | accompaniment: arpeggiated | "
     "performance: flexible connected | tempo: fast",
     "A gentle waltz turns in the quiet air, like a carousel spinning "
     "slowly under soft twilight, carrying memories of a moving castle "
     "through a peaceful, dreamlike haze.",
     "Paper lanterns loosed over a night river, each one wobbling upward on "
     "its own small current, the water beneath rippling with quick, "
     "weightless circles of light."),
    ("emotion: sad | genre: pop | accompaniment: sustained | "
     "performance: flexible connected | tempo: fast",
     "Lost in a cold, unfamiliar arena, the protagonist clutches a fragile "
     "hope while shadows of deception and survival close in from all sides.",
     "Running through airport corridors long after the gate has closed, "
     "clutching a boarding pass for a life that already left, announcements "
     "echoing overhead in a language grief almost understands."),
    ("emotion: calm | genre: jazz | accompaniment: chordal | "
     "performance: steady connected | tempo: slow",
     "A dimly lit lounge after midnight, smoke curling above half-empty "
     "glasses as unhurried chords settle over the last few listeners like "
     "a familiar coat.",
     "Sunday laundry swaying on a fire-escape line, the city idling below "
     "in warm haze, the afternoon landing as easily as light on the brick "
     "across the street."),
]

# Concrete seed elements, two sampled per piece (deterministic in piece_id),
# to decorrelate within-domain stereotypes.
SEEDS = [
    "rain", "snow", "wind", "fog", "sunrise", "sunset", "moonlight",
    "starlight", "clouds", "thunder", "rainbow", "frost", "dew", "puddle",
    "river", "stream", "lake", "sea", "waves", "beach", "island", "harbor",
    "boat", "sail", "fishing net", "lighthouse", "bridge", "mountain",
    "hill", "valley", "meadow", "field", "forest", "pine tree", "oak tree",
    "willow", "orchard", "garden", "flower bed", "roses", "sunflower",
    "wildflowers", "grass", "fallen leaves", "moss", "mushroom", "acorn",
    "pinecone", "seeds", "harvest", "wheat", "vineyard", "greenhouse",
    "well", "windmill", "barn", "farmhouse", "fence", "gate", "stone wall",
    "path", "trail", "footprints", "crossroads", "street", "alley",
    "square", "market", "fountain", "bench", "streetlamp", "balcony",
    "rooftop", "chimney", "doorway", "porch", "stairs", "window", "curtain",
    "kitchen", "bread", "soup", "tea", "coffee", "honey", "apples",
    "oranges", "grapes", "picnic", "candle", "fireplace", "blanket",
    "pillow", "bed", "cradle", "rocking chair", "bookshelf", "book",
    "letter", "postcard", "photograph", "diary", "pencil", "paintbrush",
    "canvas", "clay", "knitting", "quilt", "sewing kit", "scissors",
    "thread", "button", "mirror", "clock", "key", "lantern", "umbrella",
    "raincoat", "boots", "scarf", "gloves", "hat", "suitcase", "backpack",
    "map", "compass", "bicycle", "train", "bus", "tram", "ferry", "car",
    "road trip", "gas station", "platform", "ticket", "playground",
    "swing", "slide", "seesaw", "sandbox", "kite", "balloon", "marbles",
    "paper boat", "paper plane", "chalk drawing", "hopscotch", "jump rope",
    "football", "fishing rod", "campfire", "tent", "sleeping bag",
    "flashlight", "stars", "school", "classroom", "chalkboard", "desk",
    "library", "museum", "church", "bell", "wedding", "birthday cake",
    "gift", "confetti", "parade", "fair", "carousel", "ferris wheel",
    "fireworks", "bakery", "florist", "barbershop", "workshop", "hammer",
    "nails", "sawdust", "ladder", "paint bucket", "garden hose",
    "watering can", "clothesline", "laundry", "broom", "basket", "jar",
    "bottle", "cup", "plate", "spoon", "kettle", "oven", "dog", "cat",
    "puppy", "kitten", "horse", "sheep", "cow", "goat", "rabbit", "deer",
    "fox", "squirrel", "hedgehog", "owl", "sparrow", "swallow", "pigeon",
    "seagull", "duck", "goose", "swan", "rooster", "hen", "butterfly",
    "bee", "ladybug", "firefly", "snail", "frog", "goldfish", "turtle",
    "spider web", "bird nest", "feather", "seashell", "pebble", "driftwood",
    "sandcastle", "tide", "starfish", "coral", "iceberg", "icicle",
    "snowman", "sled", "ice skates", "autumn", "spring", "summer",
    "winter", "morning", "noon", "dusk", "midnight", "shadow", "echo",
    "silence", "breeze", "heat", "cold", "warmth",
]

ATTRIBUTE_FIELDS = ("emotion", "genre", "accompaniment", "performance",
                    "tempo")


def caption_violations(text: str) -> list[str]:
    """Return human-readable rule violations; empty means the caption is valid."""
    violations = []
    words = MUSIC.findall(text)
    if words:
        found = ", ".join(sorted({w.lower() for w in words}))
        violations.append(f"{MUSIC_VIOLATION}: {found}")
    if CJK.search(text):
        violations.append(f"{CJK_VIOLATION} (must be plain English)")
    count = len(text.split())
    if count < MIN_WORDS:
        violations.append(f"{SHORT_VIOLATION}: {count} words, minimum {MIN_WORDS}")
    return violations


def attributes_line(row: dict) -> str:
    parts = [f"{f}: {row[f]}" for f in ATTRIBUTE_FIELDS if str(row[f]).strip()]
    return " | ".join(parts)


def user_message(attrs: str, caption: str, domain: str | None = None,
                 seeds: list[str] | None = None) -> str:
    hint = f"\nScene-domain suggestion: {domain}" if domain else ""
    if seeds:
        hint += f"\nConcrete elements you may weave in: {', '.join(seeds)}"
    return f"Attributes: {attrs}\nCurrent caption: {caption}{hint}\nRewrite:"


def build_messages(row: dict) -> list[dict]:
    digest = hashlib.sha1(row["piece_id"].encode()).digest()
    rng = random.Random(digest)
    shots = rng.sample(FEWSHOT, FEWSHOT_PER_PIECE)
    domain = rng.choice(DOMAINS)
    seeds = rng.sample(SEEDS, 2)
    messages = [{"role": "system", "content": SYSTEM}]
    for attrs, original, rewrite in shots:
        messages.append({"role": "user",
                         "content": user_message(attrs, original)})
        messages.append({"role": "assistant", "content": rewrite})
    messages.append({"role": "user",
                     "content": user_message(attributes_line(row),
                                             row["imagery"], domain,
                                             seeds)})
    return messages


def retry_message(violations: list[str]) -> str:
    return (
        "That caption breaks the rules — "
        + "; ".join(violations)
        + ". Write a completely fresh caption for the same attributes and "
        "mood that obeys every rule: pure scenery in plain English, no "
        f"musical or instrument words at all, at least {MIN_WORDS} words. "
        "Output ONLY the caption."
    )


def load_rows() -> list[dict]:
    columns = ["piece_id", "imagery", *ATTRIBUTE_FIELDS]
    data = pq.read_table(LABELS, columns=columns).to_pydict()
    return [{c: str(data[c][i]) for c in columns}
            for i in range(len(data["piece_id"]))]


async def rewrite_all(rows: list[dict], out_path: Path,
                      ports: list[int]) -> None:
    cached: dict[str, str] = {}
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                record = json.loads(line)
                cached[record["piece_id"]] = record["rewrite"]
    valid = {p: text for p, text in cached.items() if not caption_violations(text)}
    requeued = len(cached) - len(valid)
    todo = [r for r in rows if r["piece_id"] not in valid]
    print(f"{len(rows)} rows, {len(valid)} cached valid, "
          f"{requeued} cached invalid re-queued, "
          f"{len(todo) - requeued} new, {len(todo)} to rewrite")

    clients = [AsyncOpenAI(base_url=f"http://localhost:{p}/v1",
                           api_key="none") for p in ports]
    semaphore = asyncio.Semaphore(CONCURRENCY * len(clients))
    lock = asyncio.Lock()
    out = out_path.open("a")
    results = dict(valid)
    failures: list[str] = []
    completed = 0

    async def one(index: int, row: dict) -> None:
        nonlocal completed
        client = clients[index % len(clients)]
        messages = build_messages(row)
        text = ""
        for attempt in range(MAX_ATTEMPTS):
            temperature = TEMPERATURE if attempt < MAX_ATTEMPTS - 2 else 0.7
            async with semaphore:
                response = await client.chat.completions.create(
                    model=MODEL, messages=messages,
                    temperature=temperature, top_p=TOP_P,
                    max_tokens=MAX_TOKENS,
                    # hybrid-reasoning model: thinking is disabled so the
                    # response is the caption itself
                    extra_body={"chat_template_kwargs":
                                {"enable_thinking": False}})
            text = response.choices[0].message.content.strip().strip('"')
            violations = caption_violations(text)
            if not violations:
                break
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": retry_message(violations)},
            ]
        else:
            failures.append(row["piece_id"])
        async with lock:
            results[row["piece_id"]] = text
            out.write(json.dumps({"piece_id": row["piece_id"],
                                  "rewrite": text}, ensure_ascii=False) + "\n")
            completed += 1
            if completed % 2000 == 0:
                out.flush()
                print(f"  {completed}/{len(todo)}")

    await asyncio.gather(*(one(i, r) for i, r in enumerate(todo)))
    out.close()

    # Collapse the append log so each piece_id appears exactly once.
    order = list(dict.fromkeys(r["piece_id"] for r in rows))
    temporary = out_path.with_name(out_path.name + ".tmp")
    with temporary.open("w") as handle:
        for piece_id in order:
            handle.write(json.dumps({"piece_id": piece_id,
                                     "rewrite": results[piece_id]},
                                    ensure_ascii=False) + "\n")
    temporary.replace(out_path)

    failure_path = out_path.parent / "rewrite_failures.json"
    failure_path.write_text(json.dumps(sorted(failures), indent=2) + "\n")
    print(f"{len(failures)} pieces never produced a clean caption: "
          f"{sorted(failures)[:10]} -> {failure_path}")


def load_rewrites() -> dict[str, str]:
    with REWRITES.open() as f:
        return {r["piece_id"]: r["rewrite"]
                for r in (json.loads(line) for line in f)}


def diversity(captions: list[str]) -> dict:
    counts = Counter(captions)
    words = [c.split() for c in captions]
    lengths = sorted(len(w) for w in words)
    unigrams, bigrams, total_uni, total_bi = set(), set(), 0, 0
    openings = set()
    for w in words:
        lower = [t.lower() for t in w]
        unigrams.update(lower)
        total_uni += len(lower)
        pairs = list(zip(lower, lower[1:]))
        bigrams.update(pairs)
        total_bi += len(pairs)
        openings.add(" ".join(lower[:4]))
    return {
        "captions": len(captions),
        "unique_captions": len(counts),
        "unique_fraction": len(counts) / len(captions),
        "most_repeated_count": counts.most_common(1)[0][1],
        "words_p50": lengths[len(lengths) // 2],
        "words_mean": round(sum(lengths) / len(lengths), 2),
        "distinct_1": round(len(unigrams) / total_uni, 5),
        "distinct_2": round(len(bigrams) / total_bi, 5),
        "unique_4word_openings": len(openings),
    }


def verify(rewrites: dict[str, str]) -> dict:
    music_hits, short, cjk = [], [], []
    for piece_id, text in rewrites.items():
        for violation in caption_violations(text):
            if violation.startswith(MUSIC_VIOLATION):
                music_hits.append(piece_id)
            elif violation.startswith(CJK_VIOLATION):
                cjk.append(piece_id)
            else:
                short.append(piece_id)
    counts = Counter(rewrites.values())
    duplicated = {c: n for c, n in counts.items() if n > 1}
    return {
        "rows": len(rewrites),
        "music_term_hits": len(music_hits),
        "music_term_ids": music_hits,
        "music_term_captions": {p: rewrites[p] for p in music_hits[:20]},
        "short_captions": len(short),
        "short_ids": short[:20],
        "cjk_captions": len(cjk),
        "cjk_ids": cjk[:20],
        "exact_duplicate_captions": len(duplicated),
        "rows_in_duplicate_groups": sum(duplicated.values()),
        "unique_captions": len(counts),
    }


def merge(rewrites: dict[str, str], backup_suffix: str) -> dict:
    table = pq.read_table(LABELS)
    ids = table["piece_id"].to_pylist()
    missing = [p for p in ids if p not in rewrites]
    if missing:
        raise ValueError(f"{len(missing)} pieces have no rewrite, "
                         f"first: {missing[:5]}")

    backup = BASE / f"labels_pre_repair_merge_{backup_suffix}.parquet"
    if backup.exists():
        raise FileExistsError(backup)
    pq.write_table(table, backup, compression="zstd")

    column = table.schema.get_field_index("imagery")
    merged = table.set_column(
        column, "imagery",
        pa.array([rewrites[p] for p in ids], type=pa.string()))
    temporary = LABELS.with_name(LABELS.name + ".tmp")
    pq.write_table(merged, temporary, compression="zstd")
    temporary.replace(LABELS)

    after = pq.read_table(LABELS)
    unchanged = all(after[name].equals(table[name])
                    for name in table.schema.names if name != "imagery")
    return {
        "backup": str(backup),
        "rows": after.num_rows,
        "other_columns_identical": unchanged,
        "imagery_matches_rewrites": (
            after["imagery"].to_pylist()
            == [rewrites[p] for p in after["piece_id"].to_pylist()]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rewrite_parser = sub.add_parser("rewrite")
    rewrite_parser.add_argument("--ports", type=str, default="8100",
                                help="comma-separated vLLM server ports")
    sub.add_parser("verify")
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--backup-suffix", required=True)
    args = parser.parse_args()

    if args.command == "rewrite":
        OUT_DIR.mkdir(exist_ok=True)
        rows = load_rows()
        ports = [int(p) for p in args.ports.split(",")]
        asyncio.run(rewrite_all(rows, REWRITES, ports))
        print("wrote", REWRITES)
        return

    rewrites = load_rewrites()
    if args.command == "verify":
        report = verify(rewrites)
        report["diversity_rewrites"] = diversity(list(rewrites.values()))
        report["diversity_labels_parquet"] = diversity(
            pq.read_table(LABELS, columns=["imagery"])["imagery"]
            .to_pylist())
    else:
        report = merge(rewrites, args.backup_suffix)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
