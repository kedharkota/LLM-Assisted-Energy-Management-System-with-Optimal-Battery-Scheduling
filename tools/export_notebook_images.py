"""Export saved notebook PNG outputs without executing notebook code."""
import argparse
import base64
import json
from pathlib import Path


def export(notebook, output_dir):
    parsed = json.loads(notebook.read_text(encoding='utf-8'))
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    stem = ''.join(c if c.isalnum() or c in '-_' else '-' for c in notebook.stem)
    for cell_index, cell in enumerate(parsed.get('cells', []), 1):
        for output_index, output in enumerate(cell.get('outputs', []), 1):
            encoded = output.get('data', {}).get('image/png')
            if not encoded:
                continue
            joined = ''.join(encoded) if isinstance(encoded, list) else encoded
            raw = base64.b64decode(''.join(joined.split()), validate=True)
            if not raw.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError(f'Invalid PNG in cell {cell_index}')
            target = output_dir / f'{stem}-cell-{cell_index}-output-{output_index}.png'
            target.write_bytes(raw)
            exported.append({'image': target.name, 'notebook': notebook.name, 'cell': cell_index, 'output': output_index})
    manifest = output_dir / f'{stem}-image-sources.json'
    manifest.write_text(json.dumps(exported, indent=2) + '\n', encoding='utf-8')
    return exported


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('docs/assets'))
    args = parser.parse_args()
    images = export(args.notebook, args.output_dir)
    print(f'Exported {len(images)} saved images. Review them before adding README links.')
