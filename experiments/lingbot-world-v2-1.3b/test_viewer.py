#!/usr/bin/env python3
"""Focused API tests for view_session.py (stdlib only, no GPU).

Run: python3 test_viewer.py
Covers: page, frame listing/since, frame serving + traversal guard,
key validation/rejection/delivery through the FIFO, log shape.
"""
import json
import os
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from wsgiref.simple_server import make_server  # noqa: F401 (kept import surface small)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import view_session as V


def png_bytes():
    # minimal valid 1x1 PNG (no PIL needed)
    def chunk(t, d):
        c = struct.pack('>I', len(d)) + t + d
        import zlib
        return c + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    import zlib
    ihdr = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
    raw = b'\x00\x00\x00\x00'
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) +
            chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


class ViewerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.frames = os.path.join(cls.tmp.name, 'frames')
        os.mkdir(cls.frames)
        for i in range(3):
            with open(os.path.join(cls.frames, 'f_%05d.png' % i), 'wb') as f:
                f.write(png_bytes())
        cls.fifo = os.path.join(cls.tmp.name, 'keys')
        os.mkfifo(cls.fifo)
        V.FRAMES = cls.frames
        V.KEYFIFO = cls.fifo
        V.ACTIONS = os.path.join(cls.tmp.name, 'actions.jsonl')
        V.KEYLOG.clear()
        import http.server
        cls.httpd = http.server.HTTPServer(('127.0.0.1', 0), V.Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.tmp.cleanup()

    def url(self, path):
        return 'http://127.0.0.1:%d%s' % (self.port, path)

    def get(self, path):
        with urllib.request.urlopen(self.url(path)) as r:
            return r.status, r.read(), dict(r.headers)

    def post(self, obj):
        req = urllib.request.Request(
            self.url('/key'), data=json.dumps(obj).encode(),
            headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_page(self):
        code, body, _ = self.get('/')
        self.assertEqual(code, 200)
        self.assertIn(b'LingBot World', body)

    def test_frames_since(self):
        code, body, _ = self.get('/frames?since=0')
        j = json.loads(body)
        self.assertEqual(j['total'], 3)
        self.assertEqual(len(j['frames']), 3)
        code, body, _ = self.get('/frames?since=3')
        self.assertEqual(json.loads(body)['frames'], [])

    def test_frame_serving_and_guards(self):
        code, body, _ = self.get('/frame/f_00001.png')
        self.assertEqual(code, 200)
        self.assertTrue(body.startswith(b'\x89PNG'))
        for bad in ['/frame/../x.png', '/frame/a.txt', '/frame/nope.png']:
            with self.assertRaises(urllib.error.HTTPError) as c:
                self.get(bad)
            self.assertEqual(c.exception.code, 404)

    def test_key_validation(self):
        code, j = self.post({'key': 'bogus'})
        self.assertEqual(code, 400)
        self.assertFalse(j['ok'])

    def test_key_rejected_without_session(self):
        code, j = self.post({'key': 'w'})
        self.assertEqual(code, 503)
        self.assertFalse(j['ok'])
        self.assertIn('session is not listening', j['error'])

    def test_key_delivery(self):
        got = []

        def reader():
            with open(self.fifo, 'r') as f:
                got.append(f.read())

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        time.sleep(0.2)
        code, j = self.post({'key': 'a'})
        t.join(timeout=5)
        self.assertEqual(code, 200)
        self.assertTrue(j['ok'])
        self.assertEqual(got, ['a\n'])

    def test_log_shape(self):
        code, body, _ = self.get('/log')
        j = json.loads(body)
        self.assertIn('keys', j)
        self.assertTrue(all(set(k) >= {'t', 'key', 'ok'} for k in j['keys']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
