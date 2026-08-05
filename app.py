import os
import re
import threading
import time
from flask import Flask, request, jsonify, render_template, send_file
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment variables")

client = genai.Client(api_key=GEMINI_API_KEY)

progress = {"value": 0, "total": 0, "status": "idle", "error": None}
output_path = "translated.srt"


def parse_srt(content):
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\Z)'
    return re.findall(pattern, content.strip())


def translate_text(text, retries=3):
    last_error = None
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=(
                    "Translate the following subtitle text to Persian. "
                    "Return only the translation, no explanation:\n" + text
                )
            )
            return response.text.strip()
        except Exception as e:
            last_error = e
            time.sleep(2 * (attempt + 1))  # backoff قبل از تلاش بعدی
    raise RuntimeError(f"Translation failed after {retries} attempts: {last_error}")


def do_translation(blocks, srt_path):
    global progress
    progress["total"] = len(blocks)
    progress["value"] = 0
    progress["status"] = "running"
    progress["error"] = None
    result = []
    try:
        for i, (num, timing, text) in enumerate(blocks):
            translated = translate_text(text.strip())
            result.append(f"{num}\n{timing}\n{translated}\n")
            progress["value"] = i + 1
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(result))
        progress["status"] = "done"
    except Exception as e:
        progress["status"] = "error"
        progress["error"] = str(e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        content = file.read().decode("utf-8")
    except UnicodeDecodeError:
        return jsonify({"error": "File encoding must be UTF-8"}), 400

    blocks = parse_srt(content)
    if not blocks:
        return jsonify({"error": "No valid SRT blocks found"}), 400

    threading.Thread(target=do_translation, args=(blocks, output_path), daemon=True).start()
    return jsonify({"status": "started", "total": len(blocks)})


@app.route("/progress")
def get_progress():
    return jsonify(progress)


@app.route("/download")
def download():
    if progress["status"] != "done":
        return jsonify({"error": "Translation not finished yet"}), 400
    return send_file(output_path, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
