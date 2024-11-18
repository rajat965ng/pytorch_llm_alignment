import os.path
import zipfile

import requests
from tqdm.auto import tqdm

if __name__ == "__main__":

    url = 'https://ideami.com/llm_align'
    file_name = url.split('/')[-1] + '.zip'
    request = requests.get(url, stream=True)
    file_size = int(request.headers.get('content-length', 0))

    if not os.path.exists(file_name):
        with tqdm.wrapattr(open(file_name, 'wb'), 'write',
                           unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                           desc=f"Downloading {file_name}", total=file_size
                           ) as fout:
            for chunk in request.iter_content(chunk_size=4096):
                fout.write(chunk)
    else:
        print(f"Already Exists: {file_name}")

    with zipfile.ZipFile(f'{file_name}', 'r') as fz:
        for file in tqdm(desc=f'Extracting {file_name}', iterable=fz.namelist(), total=len(fz.namelist())):
            fz.extract(file)

    for file in tqdm(desc=f'Deleting {file_name}', iterable=list(file_name), total=len(list(file_name))):
        os.remove(file_name)
        break
