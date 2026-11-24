"""Workspaces created before Codex was removed have a ./codex launcher that runs this module."""
import sys

MESSAGE = """This workspace was created for the Codex terminal chat, which has been removed.
The lab now uses Claude Code: create a fresh workspace with `mlab init DIR --task TASK` and run ./claude there,
or open the course page with `mlab web`. Your records in this workspace are unchanged and still readable
with `mlab history` and `mlab report`."""


def main(argv=None):
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
