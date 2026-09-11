"""CPU-only tests for the in-memory 2x present server (no GPU).

Covers latest-wins publish ordering, one-shot JPEG endpoint bytes,
MJPEG part headers, stats, and clean shutdown.
"""
import io
import json
import unittest
import urllib.request

from interactive_world import PresentServer


def _jpeg(color, size=(64, 32)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, 'JPEG', quality=80)
    return buf.getvalue()


class PresentServerTest(unittest.TestCase):
    def setUp(self):
        self.srv = PresentServer(0)
        self.srv.start()
        self.port = self.srv._httpd.server_address[1]

    def tearDown(self):
        self.srv.stop()

    def _get(self, path):
        with urllib.request.urlopen(
                f'http://127.0.0.1:{self.port}{path}',
                timeout=10) as r:
            return r.status, dict(r.headers), r.read()

    def test_empty_slot_returns_204(self):
        status, _, _ = self._get('/latest2x.jpg')
        self.assertEqual(status, 204)

    def test_publish_latest_wins_and_stale_ignored(self):
        red, blue = _jpeg((255, 0, 0)), _jpeg((0, 0, 255))
        self.srv.publish(0, red, 100.0)
        self.srv.publish(0, blue, 101.0)  # stale idx: must not overwrite
        self.srv.publish(2, blue, 102.0)
        idx, blob, t_slot = self.srv.snapshot()
        self.assertEqual(idx, 2)
        self.assertEqual(blob, blue)
        self.assertEqual(t_slot, 102.0)

    def test_latest_jpg_bytes_decode(self):
        from PIL import Image
        blue = _jpeg((0, 0, 255))
        self.srv.publish(5, blue, 200.0)
        status, headers, body = self._get('/latest2x.jpg')
        self.assertEqual(status, 200)
        self.assertEqual(headers.get('X-Frame-Idx'), '5')
        self.assertEqual(body, blue)
        px = Image.open(io.BytesIO(body))
        self.assertEqual(px.size, (64, 32))
        r, g, b = px.convert('RGB').getpixel((2, 2))
        self.assertGreater(b, 200)

    def _stats_body(self):
        with urllib.request.urlopen(
                f'http://127.0.0.1:{self.port}/present_stats',
                timeout=10) as r:
            return r.read().decode()

    def test_present_stats_reports_slot(self):
        self.srv.publish(7, _jpeg((0, 255, 0)), 300.0)
        body = json.loads(self._stats_body())
        self.assertEqual(body['idx'], 7)
        self.assertEqual(body['t_slot'], 300.0)

    def test_mjpeg_stream_first_part(self):
        import socket
        self.srv.publish(9, _jpeg((255, 255, 0)), 400.0)
        s = socket.create_connection(('127.0.0.1', self.port), timeout=10)
        try:
            s.sendall(b'GET /stream2x.mjpg HTTP/1.0\r\nHost: x\r\n\r\n')
            data = b''
            while b'X-Frame-Idx: 9' not in data:
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
                if len(data) > 1 << 20:
                    break
        finally:
            s.close()
        self.assertIn(b'X-Frame-Idx: 9', data)
        self.assertIn(b'X-Slot-Time: 400.0', data)
        self.assertIn(b'\xff\xd8', data)  # JPEG SOI present

    def test_stop_is_clean(self):
        self.srv.stop()
        idx, _, _ = self.srv.snapshot()
        self.assertEqual(idx, -1)


if __name__ == '__main__':
    unittest.main()
