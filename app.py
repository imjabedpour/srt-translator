import os, threading, re
from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

progress = {"value": 0, "total": 0}
progress_lock = threading.Lock()
output_path = "translated.srt"

def parse_srt(content):
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\Z)'
    return re.findall(pattern, content.strip())

def translate_text(text):
    prompt = f"""ترجمه دقیق زیرنویس زیر را به فارسی ارائه بده. فقط ترجمه را برگردان، بدون هیچ توضیح اضافه:

{text}

نکات مهم:
- اگر چند خط متن وجود دارد، آنها را با " ⏎ " از هم جدا کن
- فقط ترجمه فارسی را برگردان"""
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config={
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40
        }
    )
    return response.text.strip()

def do_translation(blocks, srt_path):
    global progress
    with progress_lock:
        progress["total"] = len(blocks)
        progress["value"] = 0
    
    result = []
    for i, (num, timing, text) in enumerate(blocks):
        translated = translate_text(text.strip())
        translated = translated.replace(" ⏎ ", "\n")
        result.append(f"{num}\n{timing}\n{translated}\n")
        
        with progress_lock:
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
        return jsonify({"error": "فایلی انتخاب نشده"}), 400
    
    try:
        content = file.read().decode("utf-8")
    except UnicodeDecodeError:
        return jsonify({"error": "فایل باید UTF-8 باشد"}), 400
    
    blocks = parse_srt(content)
    if not blocks:
        return jsonify({"error": "فرمت فایل SRT نامعتبر است"}), 400
    
    threading.Thread(target=do_translation, args=(blocks, output_path), daemon=True).start()
    return jsonify({"status": "started", "total": len(blocks)})

@app.route("/progress")
def get_progress():
    with progress_lock:
        return jsonify(progress)

@app.route("/download")
def download():
    if not os.path.exists(output_path):
        return jsonify({"error": "فایل ترجمه‌شده آماده نیست"}), 404
    return send_file(output_path, as_attachment=True, download_name="translated.srt")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
