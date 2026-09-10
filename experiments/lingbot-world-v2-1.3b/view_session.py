#!/usr/bin/env python3
"""Live session viewer: serves newest decoded frame with auto-refresh.

Usage: view_session.py [FRAMES_DIR] [PORT]
Defaults: /ai/outputs/lingbot-session/frames, port 8734.
"""
import functools
import http.server
import os
import sys

FRAMES = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                          '/ai/outputs/lingbot-session/frames')
PORT = int(sys.argv[2] if len(sys.argv) > 2 else 8734)

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>LingBot live view</title>
<style>html,body{margin:0;background:#000;height:100%;display:flex;
align-items:center;justify-content:center;flex-direction:column}
img{max-width:100vw;max-height:92vh;image-rendering:auto}
#s{color:#888;font:12px monospace;padding:6px}</style></head><body>
<img id="v" alt="waiting for first frame..."><div id="s"></div>
<script>
let n = 0;
async function tick() {
  try {
    const r = await fetch('/latest?n=' + (++n), {cache: 'no-store'});
    if (r.ok) {
      const b = await r.blob();
      document.getElementById('v').src = URL.createObjectURL(b);
      document.getElementById('s').textContent =
        (r.headers.get('X-Frame') || '') + '  updated ' + new Date().toLocaleTimeString();
    } else {
      document.getElementById('s').textContent = 'no frames yet — press a movement key';
    }
  } catch (e) {
    document.getElementById('s').textContent = 'viewer connection issue, retrying...';
  }
  setTimeout(tick, 800);
}
tick();
</script></body></html>""".encode('utf-8')


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = 'lingbot-view/1'

    def log_message(self, *a):
        pass

    def _latest(self):
        try:
            names = sorted(f for f in os.listdir(FRAMES) if f.endswith('.png'))
        except FileNotFoundError:
            return None, None
        if not names:
            return None, None
        return names[-1], os.path.join(FRAMES, names[-1])

    def do_GET(self):
        if self.path == '/' or self.path.startswith('/?'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path.startswith('/latest'):
            name, path = self._latest()
            if path is None:
                self.send_response(404)
                self.end_headers()
                return
            with open(path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('X-Frame', name)
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()


http.server.HTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
