"""Render the locally reviewed, offline experiment card layout."""
import argparse
import html
import json
from pathlib import Path


def render_review(manifest, output):
    output = Path(output)
    def esc(value):
        return html.escape(str(value), quote=True)
    def local(value):
        if not isinstance(value, str) or not value or ':' in value or value.startswith(('/', '//')):
            raise ValueError(f'Expected local relative path: {value!r}')
        if not (output.parent/value).is_file():
            raise FileNotFoundError(output.parent/value)
        return esc(value)
    title = esc(manifest['title'])
    parts = ['<!doctype html><html lang="zh"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<title>{title}</title>',
        '<style>body{background:#111;color:#eee;font:17px/1.6 sans-serif;max-width:1400px;margin:30px auto;padding:20px}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:20px}'
        'video,img{width:100%;height:auto}a{color:#9cf}article{background:#222;padding:15px;border-radius:6px}'
        '.status{display:inline-block;padding:3px 12px;border:1px solid #aaa;border-radius:4px}'
        '.rejected,.failed{border-color:#f88;color:#faa}.pending{border-color:#fc8;color:#fc8}'
        '.passed{border-color:#9d9;color:#9d9}.summary{border-left:4px solid #fc8;padding-left:15px}</style>',
        f'</head><body><h1>{title}</h1><p class="summary">{esc(manifest["summary"])}</p><div class="grid">']
    states={'reference':'参考','passed':'通过指定检查','failed':'失败','rejected':'已否定','pending':'待审阅','diagnostic':'仅诊断'}
    for card in manifest['cards']:
        state=card['status']
        if state not in states:
            raise ValueError(f'Unknown status: {state}')
        video=local(card['video'])
        parts += [f'<article><h2>{esc(card["title"])}</h2><p class="status {state}">{states[state]}</p>',
            f'<p>{esc(card.get("description", ""))}</p>',
            f'<video src="{video}" controls preload="metadata" playsinline></video>',
            f'<p><a href="{video}">视频文件</a></p>']
        for link in card.get('links',[]):
            parts.append(f'<p><a href="{local(link["path"])}">{esc(link["label"])}</a></p>')
        parts.append('</article>')
    parts.append('</div>')
    for picture in manifest.get('images',[]):
        label=esc(picture['label'])
        parts.append(f'<h2>{label}</h2><img src="{local(picture["path"])}" alt="{label}" loading="lazy">')
    parts.append('<nav>')
    for link in manifest.get('links',[]):
        parts.append(f'<p><a href="{local(link["path"])}">{esc(link["label"])}</a></p>')
    parts.append('</nav></body></html>')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(parts),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    render_review(json.loads(args.manifest.read_text(encoding='utf-8')),args.output)


if __name__=='__main__':
    main()
