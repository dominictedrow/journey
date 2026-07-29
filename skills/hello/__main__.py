"""Trivial skill proving the envelope round-trips end to end. Run: python -m skills.hello"""

from .._lib import envelope


def main():
    envelope.progress("hello: starting")
    sections = [
        {
            "type": "stat",
            "title": "At a glance",
            "stats": [{"label": "Status", "value": "ok"}],
        }
    ]
    envelope.progress("hello: done")
    envelope.emit("hello", ok=True, summary="Envelope round-trip succeeded", sections=sections)


if __name__ == "__main__":
    main()
