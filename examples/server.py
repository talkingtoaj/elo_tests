"""A small web service around the library, so ANY platform can use it.

Your course software (or a tool like Zapier, Make or a script) sends plain web
requests and gets plain JSON back. No Python knowledge is needed on its side.

    python examples/server.py --bands 10 --state course.json --port 8080

It listens on 127.0.0.1 only and has no password. Keep it inside your own
network. Put your own login in front of it before exposing it anywhere.

The whole conversation (see COOKBOOK.md for the same thing with curl):

    POST /questions               {"questions": [{"id": "q1", "band": 3}, ...]}
    POST /students/s42/next       {"mode": "adaptive"}  -> {"question": "q9", "estimate": {...}}
    POST /students/s42/answer     {"question": "q9", "correct": true} -> {"estimate": {...}}
    GET  /students/s42            -> {"estimate": {...}}
    POST /fit                     -> what was learned from the answers so far
    GET  /flags                   -> questions that look like they sit in the wrong band
    GET  /course                  -> the saved state

Everything is saved to the --state file after each change, and read back when
the server starts, so you can stop and restart it any time.
"""
import argparse
import json
import pathlib
import re
import sys
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from bandtest import Course  # noqa: E402


class NotFound(Exception):
    pass


class Service:
    """The course plus a lock, so two requests never change it at once."""

    def __init__(self, bands, state_path):
        self.state_path = pathlib.Path(state_path) if state_path else None
        self.lock = threading.Lock()
        if self.state_path and self.state_path.exists():
            self.course = Course.from_json(self.state_path.read_text())
        else:
            self.course = Course(bands)

    def save(self):
        if self.state_path:
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(self.course.to_json())
            tmp.replace(self.state_path)  # never leaves a half-written file

    def handle(self, method, path, body):
        c = self.course
        if method == "GET" and path == "/course":
            return json.loads(c.to_json())
        if method == "GET" and path == "/flags":
            return {"flagged": [dict(zip(("question", "listed_band", "looks_like_band", "answers"), f))
                                for f in c.flagged_questions()]}
        if method == "POST" and path == "/questions":
            for q in body.get("questions", []):
                c.add_question(str(q["id"]), int(q["band"]))
            self.save()
            return {"questions": len(c._ids)}
        if method == "POST" and path == "/fit":
            report = c.fit()
            self.save()
            return {"curve": asdict(report.curve), "answers": report.n_responses,
                    "students": report.n_students, "fitted_difficulty": report.fitted_difficulty}
        m = re.fullmatch(r"/students/([^/]+)(/next|/answer)?", path)
        if m:
            student, action = m.group(1), m.group(2)
            session = c.start_student(student, mode=body.get("mode", "adaptive"),
                                      prior_band=body.get("prior_band"))
            if method == "GET" and action is None:
                return {"estimate": asdict(session.estimate())}
            if method == "POST" and action == "/next":
                return {"question": session.next_question(), "estimate": asdict(session.estimate())}
            if method == "POST" and action == "/answer":
                session.record(str(body["question"]), bool(body["correct"]))
                self.save()
                return {"estimate": asdict(session.estimate())}
        raise NotFound(f"no such page: {method} {path}")


def make_server(service, host="127.0.0.1", port=8080):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _run(self, method):
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}") if length else {}
                with service.lock:
                    result = service.handle(method, self.path.split("?")[0], body)
                self._reply(200, result)
            except NotFound as e:
                self._reply(404, {"error": str(e)})
            except KeyError as e:
                self._reply(400, {"error": f"missing field {e}"})
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                self._reply(400, {"error": str(e)})

        def do_GET(self):
            self._run("GET")

        def do_POST(self):
            self._run("POST")

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bands", type=int, default=10, help="number of bands (used when there is no saved state)")
    ap.add_argument("--state", default="course.json", help="file where the course is saved")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    server = make_server(Service(args.bands, args.state), port=args.port)
    print(f"Listening on http://127.0.0.1:{args.port}  (state file: {args.state})")
    server.serve_forever()
