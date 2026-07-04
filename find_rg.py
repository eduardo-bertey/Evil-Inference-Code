from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq

fs = HfFileSystem()
path = 'hf://datasets/epfml/FineWeb2-HQ/spa_Latn/000_00000.parquet'
with fs.open(path, 'rb') as f:
    pf = pq.ParquetFile(f)
    md = pf.metadata
    cumul = 0
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        cumul += rg.total_byte_size
        if cumul >= 200 * 1024 * 1024:
            print(f'200MB -> RG[{i}]')
            table = pf.read_row_groups([i], columns=['text'])
            first = str(table.column('text')[0])
            print(f'text: {first[:200]}')
            break
