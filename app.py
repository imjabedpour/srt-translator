import os, threading, re
from flask import Flask, request, jsonify, render_template, send_file
from openai import OpenAI

app = Flask(__name__)

# OpenRouter configuration
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENAI_API_KEY")  # Use OpenRouter key here
)

progress = {"value": 0, "total": 0}
output_path = "translated.srt"

def parse_srt(content):
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\Z)'
    return re.findall(pattern, content.strip())

def translate_text(text):
    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct:free",
        messages=[
            {"role": "user", "content": f"Translate the following subtitle text to Persian. Return only the translation, no explanation:\n{text}"}
        ]
    )
    return response.choices[0].message.content.strip()

def do_translation(blocks, srt_path):
    global progress
    progress["total"] = len(blocks)
    progress["value"] = 0
    result = []
    for i, (num, timing, text) in enumerate(blocks):
        translated = translate_text(text.strip())
        result.append(f"{num}\n{timing}\n{translated}\n")
        progress["value"] = i + 1
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file"}), 400
    content = file.read().decode("utf-8")
    blocks = parse_srt(content)
    threading.Thread(target=do_translation, args=(blocks, output_path)).start()
    return jsonify({"status": "started", "total": len(blocks)})

@app.route("/progress")
def get_progress():
    return jsonify(progress)

@app.route("/download")
def download():
    return send_file(output_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
