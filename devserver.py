"""Local preview server.

Python's built-in server sends a large file in one write. On this machine that
write fills the socket and then waits while still holding the interpreter
lock, so the browser never receives the rest. Sending in small pieces lets
the browser read while the file is still going out.
"""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import os
import shutil

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer")


class Handler(SimpleHTTPRequestHandler):
    def copyfile(self, source, outputfile):
        shutil.copyfileobj(source, outputfile, length=64 * 1024)


if __name__ == "__main__":
    os.chdir(ROOT)
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Serving viewer at http://127.0.0.1:8765", flush=True)
    server.serve_forever()
