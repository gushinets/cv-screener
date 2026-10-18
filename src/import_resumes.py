"""Index resumes in data; repeat runs replace the same documents."""
import asyncio
from pathlib import Path

from common.store import getStore
from resumes import ResumeError, index_resume, library_files, read_resume


async def import_resumes(data_dir=Path(__file__).resolve().parents[1] / 'data'):
    files = library_files(data_dir)
    if not files:
        raise ValueError(f'No PDF, DOCX or TXT resumes found in {data_dir}')

    documents = []
    for path in files:
        try:
            documents.append((path, read_resume(path)))
        except ResumeError as error:
            raise ResumeError(f'{path.name}: {error}') from error

    store = getStore()
    try:
        for path, text in documents:
            # shortcut: renamed or moved files get new IDs; add reconciliation if file removal must sync.
            await index_resume(store, path, text)
            print(f'Indexed: {path.name}', flush=True)
        print(f'Imported {len(files)} resumes; {await store.acount_documents()} chunks.', flush=True)
    finally:
        await store._client.close()


if __name__ == '__main__':
    asyncio.run(import_resumes())
