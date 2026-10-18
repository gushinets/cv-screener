"""Structured recruiting results, grounded in the same search and source snapshots as chat."""
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

import cv_screener as core


class Output(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Fact(Output):
    text: str
    quote: str


class Candidate(Output):
    source_id: int
    name: str
    role: str
    summary: str
    facts: list[Fact]
    gaps: list[str]


class Cell(Output):
    source_id: int
    text: str
    status: Literal['confirmed', 'clarify']


class Criterion(Output):
    title: str
    cells: list[Cell]


class Analysis(Output):
    title: str
    answer: str
    candidates: list[Candidate]
    criteria: list[Criterion]
    recommendation: str
    recommended_source_id: int | None
    questions: list[str]


STRUCTURED_PROMPT = '''Верни ответ и данные для интерфейса подбора кандидатов.
answer — самостоятельный ответ на текущий вопрос с конкретными ссылками [1], [2].
Перечисляй каждый источник отдельной ссылкой: [1], [2], [3]. Не используй диапазоны ссылок.
title — краткое название подбора. candidates — максимум 5 действительно относящихся к задаче
кандидатов в порядке рекомендации; не включай нерелевантные резюме ради количества.
Ты видишь извлечённый текст, а не визуальную вёрстку файла. Не утверждай, что проверил
оформление или расположение таблиц/колонтитулов; без метаданных их расположение неизвестно.
source_id — номер резюме из текущего контекста. Имя сохраняй как в оригинале;
если имени нет, используй название файла. role, summary и facts должны опираться на резюме.
У каждого факта quote — НЕПУСТАЯ ДОСЛОВНАЯ цитата из этого же резюме без сокращений,
многоточий или перевода. Сам text пиши на языке пользователя. Не придумывай опыт.
facts — 1–5 самых существенных фактов. Каждая quote — один непрерывный фрагмент
в исходном порядке, без склеивания дат и строк из разных мест. Для нескольких периодов
используй отдельные факты или оставь подробное сопоставление в answer. Все facts карточки
цитируют её основной source_id; другие версии резюме можно сопоставить в answer со ссылками.
gaps — что требует уточнения; отсутствие упоминания не доказывает отсутствие навыка.
now/Present в резюме относится к моменту его составления и не подтверждает занятость сегодня.
Если дата актуальности неизвестна, не продолжай открытые периоды до сегодняшней даты
и не выдавай такой пересчёт за подтверждённый стаж. Приводи заявленные сроки, отдельно
оценивай закрытые периоды и отмечай расхождения. Расчёт на заданную дату допустим только
как явно условная оценка при предположении, что работа продолжалась до этой даты.
Формулировки «8+ лет», «over 5 years», «более» задают нижнюю границу, а не точный стаж.
Более длинная хронология сама по себе им не противоречит; не объявляй такое расхождением.
criteria — максимум 6 одинаковых критериев, с одной ячейкой для КАЖДОГО кандидата.
status confirmed означает, что текст подтверждён; clarify — требуется уточнение.
Перепроверяй арифметику и числовые сравнения по всем сравниваемым кандидатам.
Метрики разных проектов перечисляй как заявленные результаты. Без подтверждения
одинаковых условий измерений не называй их лучшими, худшими, наибольшими или наименьшими
и не используй как рейтинг кандидатов. Явно указывай, что сопоставимость условий требует проверки.
recommendation — обоснование выбора, questions — до 5 вопросов для интервью.
recommended_source_id — номер рекомендованного кандидата, null, если рекомендация неуместна.
При уточнении сохраняй контекст обсуждаемого кандидата. Не заменяй его первым результатом поиска.
Если подходящих кандидатов нет, candidates и criteria пусты, а answer объясняет ограничения.
В answer, recommendation, текстах фактов и ячеек используй только номера текущих источников.'''


def normalize(text):
    return ' '.join(text.split())


def validate_analysis(result, sources):
    if not core.valid_citations(result.answer, len(sources)):
        raise ValueError('В answer нужны существующие числовые ссылки на источники без диапазонов.')
    if len(result.candidates) > 5 or len(result.criteria) > 6:
        raise ValueError('Слишком много кандидатов или критериев.')
    ids = [candidate.source_id for candidate in result.candidates]
    if len(set(ids)) != len(ids) or any(not 1 <= index <= len(sources) for index in ids):
        raise ValueError('source_id кандидатов должны быть уникальными номерами текущих источников.')
    if result.recommended_source_id is not None and result.recommended_source_id not in ids:
        raise ValueError('Рекомендовать можно только кандидата из candidates.')
    for candidate in result.candidates:
        source = sources[candidate.source_id - 1]
        if normalize(candidate.name).casefold() not in normalize(source['text'] + ' ' + source['name']).casefold():
            raise ValueError('Имя кандидата должно существовать в соответствующем резюме или названии файла.')
        if not candidate.facts or len(candidate.facts) > 5:
            raise ValueError('Для каждого кандидата нужны 1–5 подтверждённых фактов.')
        for index, fact in enumerate(candidate.facts, 1):
            if not fact.quote.strip() or normalize(fact.quote) not in normalize(source['text']):
                raise ValueError(f'У кандидата {candidate.name} ([{candidate.source_id}]) факт {index}: '
                                 'quote отсутствует в этом источнике. Нужен один непрерывный дословный '
                                 'фрагмент без склеивания строк, перестановки разделов, перевода и пересказа. '
                                 'Исправь этот факт; если непрерывную цитату подобрать нельзя, замени его '
                                 'одним подтверждённым фактом, а сопоставление оставь в answer.')
    for criterion in result.criteria:
        if sorted(cell.source_id for cell in criterion.cells) != sorted(ids):
            raise ValueError('Каждый критерий должен содержать ровно одну ячейку для каждого кандидата.')
    generated = [result.answer, result.recommendation, *result.questions]
    for candidate in result.candidates:
        generated.extend([candidate.summary, *candidate.gaps, *[fact.text for fact in candidate.facts]])
    for criterion in result.criteria:
        generated.extend(cell.text for cell in criterion.cells)
    if any(core.CITATION_RANGE.search(text) for text in generated):
        raise ValueError('Не используй диапазоны ссылок. Перечисли каждый источник отдельно: [1], [2], [3].')
    for number in re.findall(r'\[(\d+)\]', '\n'.join(generated)):
        if not 1 <= int(number) <= len(sources):
            raise ValueError('В ответе обнаружена ссылка на несуществующий источник.')


def history_for_model(messages):
    history = []
    for message in messages:
        content = message['content']
        if message['role'] == 'assistant' and message.get('analysis'):
            cards = []
            for candidate in message['analysis']['candidates']:
                cards.append(f"{candidate['name']} [{candidate['source_id']}]: {candidate['summary']}\n"
                             + '\n'.join(fact['text'] for fact in candidate['facts']))
            content += '\nКарточки этого ответа:\n' + '\n'.join(cards)
        history.append({'role': message['role'], 'content': content,
                        'sources': message.get('sources', [])})
    return history


async def analyze(message, stored_messages):
    history = history_for_model(stored_messages)
    sources = await core.retrieve_sources(message, history)
    if not sources:
        return {'content': 'По этому запросу резюме не найдены. Уточните роль или навыки.',
                'sources': [], 'analysis': None, 'model': core.chat_model}
    messages = core.model_messages(message, history, sources)
    messages[0]['content'] += '\n\n' + STRUCTURED_PROMPT
    for attempt in range(2):
        response = await core.client.chat.completions.parse(
            model=core.chat_model, messages=messages, response_format=Analysis, **core.generation_parameters())
        reply = response.choices[0].message
        result = reply.parsed
        if result is None:
            raise ValueError('Модель не сформировала результат. Уточните задачу подбора.')
        try:
            validate_analysis(result, sources)
        except ValueError as error:
            if attempt:
                raise ValueError('Не удалось подтвердить цитаты и номера источников. Уточните запрос.') from error
            messages.extend([{'role': 'assistant', 'content': reply.content or result.model_dump_json()},
                             {'role': 'user', 'content': f'Исправь структуру и проверяемые данные: {error}\n'
                              f'Исходный вопрос пользователя: {message}\n'
                              'Верни самостоятельный итоговый ответ на исходный вопрос. '
                              'Не описывай внутреннюю проверку и исправление ответа.'}])
            continue
        return {'content': result.answer, 'sources': sources, 'analysis': result.model_dump(),
                'model': response.model}
