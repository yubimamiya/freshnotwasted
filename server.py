"""Run:  .venv\\Scripts\\python.exe server.py   then open http://localhost:8000"""
import json, os, webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import freshcast, agent

PLANS = {False: freshcast.build(False), True: freshcast.build(True)}
HERE = os.path.dirname(os.path.abspath(__file__))

class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=os.path.join(HERE, "site"), **k)
    def _json(self, obj):
        b = json.dumps(obj).encode(); self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", len(b)); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/plan":
            return self._json(PLANS[parse_qs(u.query).get("storm", ["0"])[0] == "1"])
        return super().do_GET()
    def do_POST(self):
        if self.path == "/api/ask":
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            d = PLANS[bool(body.get("storm"))]
            return self._json(agent.answer(body["q"], d["stores"][body["store"]], d))
        self.send_error(404)
    def log_message(self, *a): pass

if __name__ == "__main__":
    print("FreshCast running at http://localhost:8000  (Ctrl+C to stop)")
    try: webbrowser.open("http://localhost:8000")
    except Exception: pass
    ThreadingHTTPServer(("127.0.0.1", 8000), H).serve_forever()
