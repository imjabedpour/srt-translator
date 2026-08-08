import os
import re
import time
import threading
import requests
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template, send_file

app = Flask(__name__)

# ---- Config ----
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# Free-tier friendly model. Overridable via env var without touching code,
# since Google's free-tier model lineup/limits change over time.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# How many subtitle blocks to translate per API call. Bigger batches = fewer
# requests = less likely to hit the free tier's daily request cap, but too
# big risks the model losing track / truncating. 15-20 is a safe range.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "15"))

# Free tier for flash-lite is roughly 15 requests/minute -> stay safely under.
SECONDS_BETWEEN_REQUESTS = float(os.environ.get("SECONDS_BETWEEN_REQUESTS", "4.5"))

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "5"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "translated.srt")

progress = {
    "value": 0,
    "total": 0,
    "status": "idle",   # idle | running | done | error
    "error": None,
}
progress_lock = threading.Lock()


def parse_srt(content):
    """Parse SRT content into (number, timing, text) tuples.
    Normalizes Windows-style line endings first, since most SRT files
    are CRLF and the naive regex would otherwise misparse or leave
    stray \\r characters in the text.
    """
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    pattern = r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\Z)"
    return re.findall(pattern, content.strip())


def decode_srt_bytes(raw_bytes):
    """Try a handful of encodings commonly seen in SRT files in the wild."""
    for enc in ("utf-8-sig", "utf-8", "cp1256", "latin-1"):
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    # Last resort: decode with replacement so we never hard-crash on this.
    return raw_bytes.decode("utf-8", errors="replace")


def call_gemini(prompt, max_retries=3):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    # Google's rollout of the new "Auth key" (AQ.) format has behaved
    # inconsistently across accounts: some need the key as a header, some
    # need it as a ?key= query param. Try header first, and if that comes
    # back as an auth error, fall back to the query param on the same
    # attempt before giving up.
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3},
    }
    key_clean = GEMINI_API_KEY.strip()

    def _try_request():
        # 1) header-based auth
        resp = requests.post(
            GEMINI_URL,
            headers={"Content-Type": "application/json", "x-goog-api-key": key_clean},
            json=body,
            timeout=60,
        )
        if resp.status_code in (401, 403):
            # 2) fall back to query-param auth
            resp = requests.post(
                GEMINI_URL,
                headers={"Content-Type": "application/json"},
                params={"key": key_clean},
                json=body,
                timeout=60,
            )
        return resp

    delay = 3
    for attempt in range(max_retries):
        try:
            resp = _try_request()
            if resp.status_code == 429:
                raise RuntimeError("Rate limited by Gemini API (429)")
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay)
            delay *= 2


def translate_batch(blocks_batch):
    """Translate a batch of (num, timing, text) tuples in ONE API call.
    Uses [[n]] markers so we can reliably split the model's reply back
    into per-block translations, even if the model adds extra spacing.
    """
    numbered_input = "\n".join(
        f"[[{i}]]\n{text.strip()}" for i, (_, _, text) in enumerate(blocks_batch)
    )
    prompt = (
        "You are translating movie/TV subtitle lines from English to Persian (Farsi). "
        "Keep the tone natural and conversational, and preserve meaning precisely, "
        "including nuance in philosophical or emotional dialogue. "
        "Below are numbered subtitle blocks marked like [[0]], [[1]], etc. "
        "Translate each block's text to Persian. "
        "Return ONLY the same [[n]] markers followed by the Persian translation, "
        "one per block, in the same order, with nothing else added "
        "(no explanations, no extra commentary):\n\n"
        f"{numbered_input}"
    )

    reply = call_gemini(prompt)

    matches = dict(
        (int(m.group(1)), m.group(2).strip())
        for m in re.finditer(r"\[\[(\d+)\]\]\s*\n?(.*?)(?=\n?\[\[\d+\]\]|\Z)", reply, re.DOTALL)
    )

    translations = []
    for i, (_, _, text) in enumerate(blocks_batch):
        translations.append(matches.get(i, text.strip()))  # fallback: keep original if missing
    return translations


def do_translation(blocks):
    with progress_lock:
        progress["total"] = len(blocks)
        progress["value"] = 0
        progress["status"] = "running"
        progress["error"] = None

    result = []
    try:
        for start in range(0, len(blocks), BATCH_SIZE):
            batch = blocks[start:start + BATCH_SIZE]
            translated_texts = translate_batch(batch)

            for (num, timing, _), translated in zip(batch, translated_texts):
                result.append(f"{num}\n{timing}\n{translated}\n")

            with progress_lock:
                progress["value"] = min(start + len(batch), len(blocks))

            time.sleep(SECONDS_BETWEEN_REQUESTS)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(result))

        with progress_lock:
            progress["status"] = "done"

    except Exception as e:
        with progress_lock:
            progress["status"] = "error"
            progress["error"] = str(e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate():
    with progress_lock:
        if progress["status"] == "running":
            return jsonify({"error": "ترجمه‌ی دیگری در حال اجراست، صبر کنید تا تمام شود"}), 409

    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "فایلی انتخاب نشده"}), 400
    if not file.filename.lower().endswith(".srt"):
        return jsonify({"error": "فقط فایل با پسوند .srt پذیرفته می‌شود"}), 400

    content = decode_srt_bytes(file.read())
    blocks = parse_srt(content)
    if not blocks:
        return jsonify({"error": "فایل SRT معتبر نیست یا خالی است"}), 400

    # Remove any previous output so a failed/old run can't be downloaded by mistake.
    if os.path.exists(output_path):
        os.remove(output_path)

    threading.Thread(target=do_translation, args=(blocks,), daemon=True).start()
    return jsonify({"status": "started", "total": len(blocks)})


@app.route("/progress")
def get_progress():
    with progress_lock:
        return jsonify(dict(progress))


@app.route("/download")
def download():
    with progress_lock:
        status = progress["status"]
    if status != "done":
        return jsonify({"error": "ترجمه هنوز آماده نیست"}), 409
    if not os.path.exists(output_path):
        return jsonify({"error": "فایلی برای دانلود موجود نیست"}), 404
    return send_file(output_path, as_attachment=True, download_name="translated_fa.srt")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
