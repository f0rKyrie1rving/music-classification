"""Local, resumable browser helper for the frozen final blind listening review."""

import argparse
import csv
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / "data/final_blind_review.csv"
DECISIONS = {
    "yes": "盲听判断：能听出目标风格。",
    "no": "盲听判断：听不出目标风格。",
    "uncertain": "盲听判断：风格边界不确定。",
}


def load_rows(path=REVIEW):
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file)); fields = file.seek(0) or None
    if len(rows) != 40 or len({row["query_id"] for row in rows}) != 40:
        raise ValueError("Expected exactly 40 unique frozen blind-review queries.")
    if any(row["auditor_hears_label"] not in {"", *DECISIONS} for row in rows):
        raise ValueError("Blind-review sheet contains an invalid decision.")
    return rows


def save_answer(query_id, decision, reason="", path=REVIEW):
    path = Path(path); rows = load_rows(path)
    if decision not in DECISIONS:
        raise ValueError("Decision must be yes, no, or uncertain.")
    matches = [row for row in rows if row["query_id"] == str(query_id)]
    if len(matches) != 1:
        raise ValueError("Unknown blind-review query ID.")
    row = matches[0]; row["auditor_hears_label"] = decision
    row["auditor_reason"] = reason.strip() or DECISIONS[decision]
    fields = list(rows[0]); temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def next_unanswered(rows):
    return next((row for row in rows if not row["auditor_hears_label"]), None)


def page(rows, message=""):
    done = sum(bool(row["auditor_hears_label"]) for row in rows); current = next_unanswered(rows)
    if current is None:
        body = """
        <main><h1>40 / 40 已完成</h1>
        <p>答案已经逐条保存。可以关闭这个页面并返回终端。</p></main>
        """
    else:
        query_id, label = current["query_id"], escape(current["label"])
        batch_left = 10 - (done % 10)
        body = f"""
        <main>
          <p class="progress">已完成 {done} / 40 · 本组再听 {batch_left} 首即可休息</p>
          <h1>这段音乐听起来像 <strong>{label}</strong> 吗？</h1>
          <audio controls preload="metadata" src="/audio/{query_id}"></audio>
          <form method="post" action="/answer">
            <input type="hidden" name="query_id" value="{query_id}">
            <label>补充理由（可不填）<input name="reason" autocomplete="off"
              placeholder="例如：有明显合成器；更像摇滚"></label>
            <div class="buttons">
              <button name="decision" value="yes">1　像</button>
              <button name="decision" value="no">2　不像</button>
              <button name="decision" value="uncertain">3　不确定</button>
            </div>
          </form>
          <p class="hint">先听约10–15秒；拿不准时继续听。按数字键1/2/3也可提交。</p>
        </main>
        <script>
          const form = document.querySelector('form');
          document.addEventListener('keydown', event => {{
            const button = form?.querySelector(`button[value="${{
              {{'1':'yes','2':'no','3':'uncertain'}}[event.key] || ''}}"]`);
            if (button && document.activeElement.tagName !== 'INPUT') button.click();
          }});
        </script>
        """
    notice = f'<p class="notice">{escape(message)}</p>' if message else ""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Music blind review</title><style>
    :root {{ color-scheme: light dark; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    body {{ margin:0; background:#111318; color:#f4f4f5; }}
    main {{ max-width:720px; margin:10vh auto; padding:36px; background:#1c1f27;
      border:1px solid #353945; border-radius:20px; box-shadow:0 18px 60px #0008; }}
    h1 {{ font-size:30px; line-height:1.35; }} strong {{ color:#82d9c7; }}
    .progress,.hint {{ color:#aeb4c2; }} audio {{ width:100%; margin:24px 0; }}
    label {{ display:block; color:#cdd1da; }} input {{ width:100%; box-sizing:border-box;
      margin-top:8px; padding:12px; border-radius:10px; border:1px solid #4b5060; font-size:16px; }}
    .buttons {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:22px; }}
    button {{ padding:16px 10px; border:0; border-radius:12px; background:#337d70;
      color:white; font-size:17px; cursor:pointer; }} button:hover {{ background:#409989; }}
    .notice {{ max-width:720px; margin:20px auto -7vh; color:#ffbd69; }}
    @media(max-width:650px) {{ main {{ margin:0; min-height:100vh; border-radius:0; padding:24px; }}
      .buttons {{ grid-template-columns:1fr; }} }}
    </style></head><body>{notice}{body}</body></html>""".encode("utf-8")


class ReviewHandler(BaseHTTPRequestHandler):
    def send_bytes(self, content, content_type="text/html; charset=utf-8", status=HTTPStatus.OK,
                   extra_headers=None):
        self.send_response(status); self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        for key, value in (extra_headers or {}).items(): self.send_header(key, value)
        self.end_headers(); self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(page(load_rows())); return
        if parsed.path.startswith("/audio/"):
            query_id = parsed.path.removeprefix("/audio/")
            matches = [row for row in load_rows() if row["query_id"] == query_id]
            if len(matches) != 1:
                self.send_error(HTTPStatus.NOT_FOUND); return
            audio = (ROOT / matches[0]["audio_file"]).resolve()
            if not audio.is_relative_to(ROOT) or not audio.is_file():
                self.send_error(HTTPStatus.NOT_FOUND); return
            data = audio.read_bytes(); start, end = 0, len(data) - 1
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                bounds = range_header[6:].split("-", 1)
                start = int(bounds[0] or 0); end = int(bounds[1] or end)
                end = min(end, len(data) - 1); data = data[start:end + 1]
                headers = {"Accept-Ranges": "bytes",
                           "Content-Range": f"bytes {start}-{end}/{audio.stat().st_size}"}
                self.send_bytes(data, "audio/wav", HTTPStatus.PARTIAL_CONTENT, headers); return
            self.send_bytes(data, "audio/wav", extra_headers={"Accept-Ranges": "bytes"}); return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if urlparse(self.path).path != "/answer":
            self.send_error(HTTPStatus.NOT_FOUND); return
        length = int(self.headers.get("Content-Length", 0))
        form = parse_qs(self.rfile.read(length).decode("utf-8"))
        try:
            save_answer(form.get("query_id", [""])[0], form.get("decision", [""])[0],
                        form.get("reason", [""])[0])
            content = page(load_rows(), "已保存，下一首已准备好。")
            self.send_bytes(content)
        except (ValueError, OSError) as error:
            self.send_bytes(page(load_rows(), str(error)), status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(); server = ThreadingHTTPServer(("127.0.0.1", args.port), ReviewHandler)
    print(f"Blind review: http://127.0.0.1:{args.port}", flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
