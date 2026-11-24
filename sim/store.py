"""What the three stores of people (faces: people.Registry, voices: speaker.Voiceprints, profiles: profiles.Profiles) have in common: how a file is written and how its log is read.

The log is the record of who was enrolled, who was forgotten and when (an audit trail the household can read). It is kept across restarts, and 'forget Tom' takes Tom's name out of
all of it, the part from earlier sessions included.
"""
import json
import os

LOG_MAX = 2000                   # entries kept: the log grows with every enrolment, so it is cut at the oldest end


def dump_json(path, obj):
    """Write `obj` to `path` as JSON, all of it or none of it: to a temporary file first, then moved over the old one. A robot's power can go at any moment; a file cut off half-way would read as an
    empty store and everybody in it would be lost."""
    tmp = f'{path}.tmp'
    try:
        mode = os.stat(path).st_mode & 0o777 if os.path.exists(path) else None             # whoever made the file readable by the owner only keeps it so
        with open(tmp, 'w') as f: json.dump(obj, f, ensure_ascii=False)
        if mode is not None: os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try: os.unlink(tmp)
        except OSError: pass
        raise


def load_log(d):
    """The log in a loaded file: a list of entries (time, event, name or None, ...), or nothing if it is missing or not that shape."""
    log = d.get('log') if isinstance(d, dict) else None
    if not isinstance(log, list): return []
    return [e for e in log if isinstance(e, (list, tuple)) and len(e) >= 2][-LOG_MAX:]
