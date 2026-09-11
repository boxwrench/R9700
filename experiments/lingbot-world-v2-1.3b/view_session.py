#!/usr/bin/env python3
"""LingBot single-window play view.

One browser window: the world, a keypress log, and keyboard input that
drives the session. Start the session with its stdin attached to the key
FIFO, then press movement keys in this page.

Usage: view_session.py [FRAMES_DIR] [PORT] [KEY_FIFO]
Defaults: /ai/outputs/lingbot-session/frames, 8734, /tmp/lingbot-keys
Session side: run_product.sh < /tmp/lingbot-keys
"""
import errno
import json
import os
import sys
import time

import http.server

FRAMES = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                          '/ai/outputs/lingbot-session/frames')
PORT = int(sys.argv[2] if len(sys.argv) > 2 else 8734)
KEYFIFO = sys.argv[3] if len(sys.argv) > 3 else '/tmp/lingbot-keys'
# Optional live 2x MJPEG stream served by the session's present server
# (interactive_world.py --upscale2 + --present_port). Empty (default) keeps
# the legacy PNG-poll display byte-for-byte as it was.
PRESENT = (sys.argv[4] if len(sys.argv) > 4
           else os.environ.get('PRESENT_URL', ''))
ACTIONS = os.path.join(os.path.dirname(FRAMES), 'actions.jsonl')

KEYMAP = {
    'w': 'w', 'a': 'a', 's': 's', 'd': 'd', 'x': 'stay',
    'arrowup': 'w', 'arrowdown': 's', 'arrowleft': 'a', 'arrowright': 'd',
}
BUTTONS = {'turn_left': 'turn_left', 'stay': 'stay', 'turn_right': 'turn_right'}

KEYLOG = []


def send_key(line):
    try:
        fd = os.open(KEYFIFO, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as e:
        if e.errno == errno.ENXIO:
            return False, 'session is not listening (start it with < $KEYFIFO)'
        return False, 'key fifo error: %s' % e
    try:
        os.write(fd, (line + '\n').encode())
    except OSError as e:
        return False, 'session is not listening (restart it?)'
    finally:
        os.close(fd)
    return True, ''


def last_action():
    try:
        with open(ACTIONS, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 2048))
            lines = f.read().decode().strip().split('\n')
        r = json.loads(lines[-1])
        return {'action': r.get('action'), 'ms': round(r.get('first_frame_latency', 0) * 1000),
                'kv': '%d/%d' % (r.get('kv_local_end', 0), r.get('kv_capacity_tokens', 0)),
                'n': len(lines)}
    except (OSError, ValueError, IndexError):
        return None


PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LingBot World</title>
<style>
  :root { color-scheme: dark; }
  html,body { margin:0; height:100%; background:#0b0c0d; overflow:hidden; }
  body { font-family:ui-sans-serif,system-ui,"Segoe UI",Helvetica,sans-serif; color:#e8e8e8; }
  #stage { position:fixed; inset:0; display:flex; align-items:center; justify-content:center; }
  #stage img { position:absolute; max-width:100vw; max-height:100vh; object-fit:contain;
               opacity:0; }
  #stage img.on { opacity:1; }
  @media (prefers-reduced-motion: no-preference) {
    #stage img { transition:opacity .12s linear; }
  }
  #hud { position:fixed; top:10px; left:12px; font-size:13px; line-height:1.7;
         background:rgba(10,10,10,.62); padding:8px 12px; border:1px solid #2a2a2a; }
  #hud .k { color:#9fd08a; font-family:ui-monospace,Menlo,monospace; }
  #hud .dim { color:#8a8a8a; }
  #bar { position:fixed; left:0; right:0; bottom:0; display:flex; gap:8px;
         align-items:center; padding:8px 12px; font-size:13px;
         background:rgba(10,10,10,.78); border-top:1px solid #2a2a2a; }
  #bar button { background:#1c1e20; color:#e8e8e8; border:1px solid #3a3a3a;
                padding:6px 14px; font-size:13px; cursor:pointer; }
  #bar button:hover { background:#2a2d30; }
  #bar button.danger { color:#e08a8a; }
  #stat { margin-left:auto; color:#8a8a8a; font-family:ui-monospace,Menlo,monospace;
          font-size:12px; white-space:nowrap; }
  #wait { position:fixed; top:12px; right:14px; font-size:13px; color:#8a8a8a;
          display:none; }
  #banner { position:fixed; top:0; left:0; right:0; display:none;
            background:#3a1414; color:#f0b0b0; font-size:13px; padding:8px 12px;
            border-bottom:1px solid #6a2a2a; z-index:10; }
  #banner.show { display:block; }
  #banner code { font-family:ui-monospace,Menlo,monospace; color:#fff; }
  @media (prefers-reduced-motion: no-preference) {
    #wait.show { display:block; animation:blink 1s steps(2) infinite; }
  }
  #wait.show { display:block; }
  @keyframes blink { 50% { opacity:.35; } }
</style></head><body>
<div id="banner"></div>
<div id="stage"><img id="a" alt=""><img id="b" alt=""><img id="live" alt="" style="display:none"></div>
<div id="hud"><span class="dim">keys —</span> <span id="keys"></span></div>
<div id="wait">generating&hellip;</div>
<div id="bar">
  <button data-k="turn_left">&larr; turn</button>
  <button data-k="stay">stay</button>
  <button data-k="turn_right">turn &rarr;</button>
  <button data-k="quit" class="danger" id="quit">quit</button>
  <span id="stat">move: WASD / arrows &middot; click the page first so keys register</span>
</div>
<script>
const STREAM = '__STREAM2X__';
const A = document.getElementById('a'), B = document.getElementById('b');
const LIVE = document.getElementById('live');
if (STREAM) {
  // Live 2x MJPEG push: newest frame wins, no filesystem polling.
  A.style.display = 'none'; B.style.display = 'none';
  LIVE.style.display = 'block'; LIVE.style.opacity = '1';
  LIVE.onerror = () => { setTimeout(() => { LIVE.src = STREAM; }, 1000); };
  LIVE.src = STREAM;
}
let shown = A, hidden = B, seenCount = -1, queue = [], busy = false;
let waiting = false, lastFrame = '';
const keysEl = document.getElementById('keys');
const statEl = document.getElementById('stat');
const waitEl = document.getElementById('wait');
const keyHist = [];

function noteKey(k, ok) {
  const t = new Date().toLocaleTimeString();
  keyHist.push('<span class="k">' + k + '</span> <span class="dim">' + t + (ok ? '' : ' ?') + '</span>');
  while (keyHist.length > 6) keyHist.shift();
  keysEl.innerHTML = keyHist.join('<br>');
}

const bannerEl = document.getElementById('banner');
function showBanner(msg) {
  bannerEl.innerHTML = msg;
  bannerEl.classList.add('show');
}
async function send(cmd) {
  noteKey(cmd, true);
  waiting = true; waitEl.classList.add('show');
  try {
    const r = await fetch('/key', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({key: cmd})});
    if (!r.ok) {
      noteKey(cmd + ' rejected', false);
      let why = '';
      try { why = (await r.json()).error || ''; } catch (e) {}
      showBanner('Key <b>' + cmd + '</b> was rejected: ' + why +
        ' — quit this session and restart it with ' +
        '<code>run_product.sh &lt; /tmp/lingbot-keys</code> so the page can drive it.');
    } else {
      bannerEl.classList.remove('show');
    }
  } catch (e) { noteKey(cmd + ' failed', false); waiting = false; waitEl.classList.remove('show'); }
}

function playNext() {
  if (busy || !queue.length) return;
  busy = true;
  const name = queue.shift();
  hidden.onload = () => {
    hidden.classList.add('on'); shown.classList.remove('on');
    [shown, hidden] = [hidden, shown];
    lastFrame = name; waiting = false; waitEl.classList.remove('show');
    setTimeout(() => { busy = false; setTimeout(playNext, 140); }, 60);
  };
  hidden.onerror = () => { busy = false; setTimeout(playNext, 300); };
  hidden.src = '/frame/' + name + '?t=' + Date.now();
}

async function poll() {
  try {
    if (!STREAM) {
      const r = await fetch('/frames?since=' + Math.max(seenCount, 0), {cache: 'no-store'});
      if (r.ok) {
        const j = await r.json();
        if (seenCount < 0) {
          // First poll: snap to the live edge. Replaying the whole backlog
          // looks like the world moving on its own; only new frames queue.
          seenCount = j.total;
        } else {
          for (const n of j.frames) { queue.push(n); seenCount++; }
          playNext();
        }
      }
    }
    const l = await (await fetch('/log', {cache: 'no-store'})).json();
    if (l.action) statEl.textContent =
      l.action.action + ' ' + l.action.ms + 'ms  kv ' + l.action.kv + '  |  WASD / arrows';
  } catch (e) {}
  setTimeout(poll, 500);
}

const MAP = {w:'w',a:'a',s:'s',d:'d',x:'stay',
  arrowup:'w',arrowdown:'s',arrowleft:'a',arrowright:'d'};
document.addEventListener('keydown', (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = MAP[e.key.toLowerCase()];
  if (k) { e.preventDefault(); send(k); }
});
document.querySelectorAll('#bar button').forEach((b) => {
  b.addEventListener('click', () => {
    if (b.id === 'quit' && !confirm('End the session?')) return;
    send(b.dataset.k);
  });
});
poll();
</script></body></html>
""".encode('utf-8')


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = 'lingbot-view/2'

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/' or self.path.startswith('/?'):
            page = PAGE.replace(b'__STREAM2X__', PRESENT.encode())
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        elif self.path.startswith('/frames'):
            try:
                since = int(self.path.split('since=')[1].split('&')[0])
            except (IndexError, ValueError):
                since = 0
            try:
                names = sorted(f for f in os.listdir(FRAMES) if f.endswith('.png'))
            except FileNotFoundError:
                names = []
            self._json({'frames': names[since:], 'total': len(names)})
        elif self.path.startswith('/frame/'):
            name = self.path[len('/frame/'):].split('?')[0]
            if '/' in name or not name.endswith('.png'):
                self.send_response(404)
                self.end_headers()
                return
            path = os.path.join(FRAMES, name)
            if not os.path.isfile(path):
                self.send_response(404)
                self.end_headers()
                return
            with open(path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/log':
            self._json({'keys': KEYLOG[-12:], 'action': last_action()})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != '/key':
            self.send_response(404)
            self.end_headers()
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
        except (ValueError, TypeError):
            self._json({'ok': False, 'error': 'bad json'}, 400)
            return
        key = str(body.get('key', '')).lower()
        cmd = KEYMAP.get(key)
        if key == 'quit':
            cmd = 'quit'
        elif key in BUTTONS:
            cmd = BUTTONS[key]
        if cmd is None:
            self._json({'ok': False, 'error': 'unknown key'}, 400)
            return
        ok, err = send_key(cmd)
        KEYLOG.append({'t': time.strftime('%H:%M:%S'), 'key': cmd, 'ok': ok})
        del KEYLOG[:-12]
        if not ok:
            self._json({'ok': False, 'error': err}, 503)
        else:
            self._json({'ok': True})


if __name__ == '__main__':
    http.server.HTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
