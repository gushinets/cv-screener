"""Read resume files and index them with the same identity as the command-line importer."""
from io import BytesIO
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from zipfile import ZipFile

from docx import Document
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

EXTENSIONS = {'.txt', '.pdf', '.docx'}
MAX_BYTES = 10 * 1024 * 1024
MAX_TEXT = 200_000


class ResumeError(ValueError):
    pass


def validate_name(name):
    if (not name or len(name) > 180 or name != name.strip() or name.endswith('.')
            or any(ord(c) < 32 or c in '<>:"/\\|?*' for c in name)
            or name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                                            *[f'COM{i}' for i in range(1, 10)],
                                            *[f'LPT{i}' for i in range(1, 10)]}):
        raise ResumeError('Недопустимое имя файла. Переименуйте резюме и повторите загрузку.')
    if Path(name).suffix.lower() not in EXTENSIONS:
        raise ResumeError('Поддерживаются PDF, DOCX и TXT.')
    return name


def library_files(folder):
    return sorted((p for p in folder.iterdir() if p.is_file() and not p.is_symlink()
                   and p.suffix.lower() in EXTENSIONS and p.resolve().parent == folder.resolve()),
                  key=lambda p: p.name.casefold())


def _word_blocks(container):
    for block in container.iter_inner_content():
        if isinstance(block, Paragraph):
            yield block.text
            # Paragraph.text omits text inside Word drawing/text-box containers.
            for paragraph in block._p.xpath('.//w:txbxContent//w:p'):
                yield Paragraph(paragraph, block._parent).text
        else:
            for row in block.rows:
                yield '\t'.join('\n'.join(_word_blocks(cell)) for cell in row.cells)


def extract_text(content, suffix):
    if not content:
        raise ResumeError('Файл пуст. Выберите резюме с текстом.')
    if len(content) > MAX_BYTES:
        raise ResumeError('Максимальный размер резюме — 10 МБ.')
    try:
        if suffix.lower() == '.pdf':
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise ResumeError('PDF защищён паролем. Загрузите копию без защиты.')
            if len(reader.pages) > 100:
                raise ResumeError('В PDF больше 100 страниц. Загрузите только резюме.')
            parts = []
            for page in reader.pages:
                stream = page.get_contents()
                if stream is not None and len(stream.get_data()) > 10 * 1024 * 1024:
                    raise ResumeError('Слишком сложная страница PDF. Экспортируйте резюме в TXT или DOCX.')
                parts.append(page.extract_text() or '')
                if sum(map(len, parts)) > MAX_TEXT:
                    raise ResumeError('В резюме слишком много текста: максимум 200 000 символов.')
            text = '\n'.join(parts)
            # shortcut: image-only PDFs need OCR; add recognition when scanned resumes are required.
            if not text.strip():
                raise ResumeError('В PDF нет текстового слоя. Распознайте скан через OCR или загрузите DOCX/TXT.')
        elif suffix.lower() == '.docx':
            with ZipFile(BytesIO(content)) as archive:
                entries = archive.infolist()
                if len(entries) > 2000 or sum(item.file_size for item in entries) > 40 * 1024 * 1024:
                    raise ResumeError('DOCX слишком велик после распаковки. Загрузите только резюме.')
            document = Document(BytesIO(content))
            headers, footers, seen = [], [], set()
            for section in document.sections:
                for part, target in [(section.header, headers), (section.footer, footers)]:
                    if part.part.partname not in seen:
                        seen.add(part.part.partname)
                        target.extend(_word_blocks(part))
            text = '\n'.join([*headers, *_word_blocks(document), *footers])
        elif suffix.lower() == '.txt':
            if content.startswith((b'\xff\xfe', b'\xfe\xff')):
                text = content.decode('utf-16')
            else:
                try:
                    text = content.decode('utf-8-sig')
                except UnicodeDecodeError:
                    text = content.decode('cp1251')
        else:
            raise ResumeError('Поддерживаются PDF, DOCX и TXT.')
    except ResumeError:
        raise
    except Exception as error:
        raise ResumeError('Не удалось прочитать файл. Проверьте, что он не повреждён и соответствует своему формату.') from error
    if not text.strip() or not any(c.isalpha() for c in text):
        raise ResumeError('В файле не найден текст резюме.')
    if any(ord(c) < 32 and c not in '\n\r\t\f' for c in text):
        raise ResumeError('Файл содержит двоичные данные. Загрузите текстовый TXT, PDF или DOCX.')
    if len(text) > MAX_TEXT:
        raise ResumeError('В резюме слишком много текста: максимум 200 000 символов.')
    return text.replace('\r\n', '\n').replace('\r', '\n')


def read_resume(path):
    if path.stat().st_size > MAX_BYTES:
        raise ResumeError('Максимальный размер резюме — 10 МБ.')
    return extract_text(path.read_bytes(), path.suffix)


def resume_id(path):
    return str(uuid5(NAMESPACE_URL, path.resolve().as_uri()))


async def index_resume(store, path, text):
    await store.aadd(objectId=resume_id(path), nodeId='cv-screener', parent=path.name,
                     permissionId=-1, text=text, tables='')
