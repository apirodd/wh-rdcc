from google.colab import drive
drive.mount('/content/drive')

import os, requests

DATA_RAW = '/content/drive/MyDrive/WH_RDCC/data_raw'
os.makedirs(DATA_RAW, exist_ok=True)  # crea la cartella se non esiste

# verifica
print('Cartella:', DATA_RAW)
print('Contenuto attuale:')
for f in sorted(os.listdir(DATA_RAW)):
    gb = os.path.getsize(f'{DATA_RAW}/{f}') / 1e9
    print(f'  {f:<45} {gb:.2f} GB')
    import requests, os

DATA_RAW = '/content/drive/MyDrive/WH_RDCC/data_raw'

BASE_API = 'https://darus.uni-stuttgart.de/api/access/datafile'

FILES = {
    'dichasus-cf02.tfrecords': '/:persistentId?persistentId=doi:10.18419/DARUS-2854/14',
    'dichasus-cf03.tfrecords': '/:persistentId?persistentId=doi:10.18419/DARUS-2854/15',
    'dichasus-cf04.tfrecords': '/:persistentId?persistentId=doi:10.18419/DARUS-2854/16',
    'spec.json':               '/:persistentId?persistentId=doi:10.18419/DARUS-2854/12',
    'reftx-offsets-dichasus-cf0x.json': '/:persistentId?persistentId=doi:10.18419/DARUS-2854/13',
}

EXPECTED_GB = {
    'dichasus-cf02.tfrecords': 4.5,
    'dichasus-cf03.tfrecords': 5.5,
    'dichasus-cf04.tfrecords': 10.3,
    'spec.json': 0.0,
    'reftx-offsets-dichasus-cf0x.json': 0.0,
}

def download_with_resume(url, dest, expected_gb):
    existing = os.path.getsize(dest) if os.path.exists(dest) else 0
    expected_bytes = expected_gb * 1e9

    # Considera completo se >= 95% della dimensione attesa
    if existing >= expected_bytes * 0.95:
        print(f'  OK: {os.path.basename(dest)} ({existing/1e9:.2f} GB)')
        return

    headers = {}
    if existing > 0:
        headers['Range'] = f'bytes={existing}-'
        print(f'  Resume da {existing/1e9:.2f} GB: {os.path.basename(dest)}')
    else:
        print(f'  Download: {os.path.basename(dest)}')

    with requests.get(url, stream=True, timeout=120,
                      headers=headers) as r:
        # 206 = Partial Content (resume ok), 200 = inizio da zero
        if r.status_code == 416:
            print(f'  File già completo (server dice Range Not Satisfiable)')
            return
        r.raise_for_status()

        mode = 'ab' if existing > 0 and r.status_code == 206 else 'wb'
        downloaded = existing if mode == 'ab' else 0

        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=16*1024*1024):
                f.write(chunk)
                downloaded += len(chunk)
                print(f'\r    {downloaded/1e9:.2f} GB', end='', flush=True)

    final_size = os.path.getsize(dest)
    print(f'\n  Completato: {final_size/1e9:.2f} GB')

for fname, path in FILES.items():
    dest = f'{DATA_RAW}/{fname}'
    url  = f'{BASE_API}{path}'
    download_with_resume(url, dest, EXPECTED_GB[fname])

print('\nFile in data_raw:')
for f in sorted(os.listdir(DATA_RAW)):
    gb = os.path.getsize(f'{DATA_RAW}/{f}') / 1e9
    print(f'  {f:<45} {gb:.2f} GB')