import sys, os
sys.path.insert(0, 'rust/moe-mla')
from datasets import load_dataset
ds = load_dataset('epfml/FineWeb2-HQ', 'spa_Latn', split='train', streaming=True)
it = iter(ds)
skip = 200*1024*1024
s = 0
for item in it:
    t = item.get('text') if isinstance(item, dict) else str(item)
    s += len(t.encode('utf-8'))
    if s >= skip:
        break
print(f'skip {s} bytes')
out = 0
with open('rust/moe-mla/fw_200_201.txt', 'w', encoding='utf-8') as f:
    for item in it:
        t = item.get('text') if isinstance(item, dict) else str(item)
        tam = len(t.encode('utf-8'))
        if out + tam > 1024*1024:
            break
        f.write(t + '\n\n')
        out += tam
print(f'got {out} bytes')
with open('rust/moe-mla/fw_200_201.txt', 'r', encoding='utf-8') as f:
    c = f.read()
print(f'chars: {len(c)}')
print(c[:200])
print('---')
print(c[-200:])
