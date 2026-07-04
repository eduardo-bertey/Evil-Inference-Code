import sys, os
sys.path.insert(0, 'rust/moe-mla')
from datasets import load_dataset

ds = load_dataset('epfml/FineWeb2-HQ', 'spa_Latn', split='train', streaming=True)
it = iter(ds)

skip_target = 200 * 1024 * 1024
skipped = 0
for item in it:
    text = item.get('text') if isinstance(item, dict) else str(item)
    skipped += len(text.encode('utf-8'))
    if skipped >= skip_target:
        break
print(f'Skip done: {skipped} bytes')

target = 1 * 1024 * 1024
appended = 0
with open('rust/moe-mla/fineweb_test_block.txt', 'w', encoding='utf-8') as f:
    for item in it:
        text = item.get('text') if isinstance(item, dict) else str(item)
        tam = len(text.encode('utf-8'))
        if appended + tam > target:
            break
        f.write(text)
        f.write('\n\n')
        appended += tam
print(f'Written: {appended} bytes')
with open('rust/moe-mla/fineweb_test_block.txt', 'r', encoding='utf-8') as f:
    c = f.read()
print(f'First: {c[:100]}')
print(f'Last: {c[-100:]}')
