"""Put the page together: playground.src.html + xs_logic.js + xs_world.js + ui.js  ->  index.html (a whole page: open it through a web server, `python -m http.server` in this folder)
and dist/artifact.html (the same without the <html>/<head>/<body> wrapper, the form the Artifact tool wants). Run: python build.py"""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
rd = lambda n: open(os.path.join(HERE, n), encoding='utf-8').read()
frag = rd('playground.src.html')
for tag, name in (('/*__XS_LOGIC__*/', 'xs_logic.js'), ('/*__XS_WORLD__*/', 'xs_world.js'), ('/*__UI__*/', 'ui.js')):
    js = rd(name)
    assert '</script' not in js, name
    frag = frag.replace(tag, js)
try: commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE, capture_output=True, text=True).stdout.strip()
except Exception: commit = ''
frag = frag.replace('__COMMIT__', commit)
os.makedirs(os.path.join(HERE, 'dist'), exist_ok=True)
open(os.path.join(HERE, 'dist', 'artifact.html'), 'w', encoding='utf-8').write(frag)
page = '<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n' + frag + '\n</body>\n</html>\n'
open(os.path.join(HERE, 'index.html'), 'w', encoding='utf-8').write(page)
print(f'index.html {len(page) / 1024:.0f} KB, dist/artifact.html {len(frag) / 1024:.0f} KB')
