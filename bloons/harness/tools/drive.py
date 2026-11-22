"""Let Claude drive the game by hand: a play.py REPL behind a localhost socket.

    python tools/drive.py serve [--difficulty easy] [--hero quincy]   (keep running, game open)
    python tools/drive.py <play.py command ...>                        e.g. drive.py place dart .4 .5
    python tools/drive.py shot                                         -> logs/shot.png (half size)
    python tools/drive.py stop                                         Dom's kill switch

`stop` writes logs/STOP and shuts the server down; every later command is refused until
logs/STOP is deleted. Same as: tools\\stop.cmd
"""
import socket
import socketserver
import sys
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

STOP = ROOT / "logs" / "STOP"
PORT = 47613


def serve(argv):
    import argparse

    import cv2
    from play import Play

    from bloons.harness import BloonsHarness
    from bloons.platform import Game

    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="monkey_meadow")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--hero", default="quincy")
    a = ap.parse_args(argv)
    h = BloonsHarness(Game(), a.map, a.difficulty, hero=a.hero)

    class Drive(Play):
        def do_shot(self, line):
            """shot: save a half-size screenshot to logs/shot.png."""
            f = h.game.grab()
            img = f.content_bgr()
            cv2.imwrite(str(ROOT / "logs" / "shot.png"), cv2.resize(img, None, fx=0.5, fy=0.5))
            self.say(f"logs/shot.png ({img.shape[1]}x{img.shape[0]} -> half)")

        def do_until(self, line):
            """until <command>: retry a command every 2 s (max 4 min) until it prints 'ok'.
            Gives up early on STOP, a lost game, or 'NO' reasons other than money."""
            import time
            t0 = time.time()
            while time.time() - t0 < 240 and not STOP.exists():
                buf = StringIO()
                Drive(h, out=buf).onecmd(line)
                text = buf.getvalue()
                if text.startswith("ok"):
                    self.say(text.rstrip())
                    return
                if "not_front" in text:
                    h.game.ensure_front()
                elif not any(w in text for w in ("poor", "money", "afford", "cash")):
                    self.say("gave up: " + text.rstrip())
                    return
                if h.screen_kind() in ("defeat", "victory"):
                    self.say("game over: " + h.screen_kind())
                    return
                if getattr(h, "on_wait", None):   # e.g. banana pickup while saving up
                    h.on_wait()
                time.sleep(2)
            self.say("timeout/stop")

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            line = self.rfile.readline().decode().strip()
            out = StringIO()
            if line == "_restart":                  # Claude reloading code; not the kill switch
                out.write("server exiting for restart\n")
                self.server.stopping = True
            elif line == "stop" or STOP.exists():
                out.write("STOPPED by Dom\n")
                self.server.stopping = True
            else:
                # A side thread sends a NUL every 30 s: something on this machine resets loopback
                # sockets that stay silent for 120 s. The client strips the NULs. The command
                # itself stays on the main thread (screen capture segfaults off it).
                import threading
                done = threading.Event()

                def beat():
                    while not done.wait(30):
                        try:
                            self.wfile.write(b"\x00")
                        except OSError:
                            return
                threading.Thread(target=beat, daemon=True).start()
                try:
                    Drive(h, out=out).onecmd(line)
                finally:
                    done.set()
            self.wfile.write(out.getvalue().encode())

    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as srv:
        srv.stopping = False
        print(f"drive server on {PORT}", flush=True)
        while not srv.stopping:
            srv.handle_request()


def send(line):
    with socket.create_connection(("127.0.0.1", PORT), timeout=600) as s:
        s.sendall((line + "\n").encode())
        s.shutdown(socket.SHUT_WR)
        return b"".join(iter(lambda: s.recv(65536), b"")).replace(b"\x00", b"").decode()


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    if args[0] == "serve":
        if STOP.exists():
            sys.exit("STOPPED by Dom (logs/STOP exists): Dom deletes it to allow driving again.")
        return serve(args[1:])
    if args[0] == "stop":
        STOP.parent.mkdir(exist_ok=True)
        STOP.write_text("stop")
        try:
            send("stop")
        except OSError:
            pass
        print("Stopped: Claude's game commands are refused until logs/STOP is deleted.")
        return
    if STOP.exists():
        print("STOPPED by Dom (logs/STOP exists). Do not continue.")
        sys.exit(1)
    print(send(" ".join(args)), end="")


if __name__ == "__main__":
    main()
