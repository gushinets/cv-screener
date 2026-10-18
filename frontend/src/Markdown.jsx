import React from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {citationPlugin, sourceAt} from './api.js';

const SourceContext = React.createContext(null);
function MarkdownLink({href, children}) {
  const {message, onSource} = React.useContext(SourceContext);
  const match = /^#cv-source-(\d+)$/.exec(href || '');
  return match ? <Citation message={message} number={Number(match[1])} onSource={onSource} />
    : <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
}
const components = {
  h1: 'h3', h2: 'h3', h3: 'h4', img: () => null, a: MarkdownLink,
  table: ({node, ...props}) => <div className="matrix-scroll" tabIndex={0} role="region" aria-label="Таблица ответа"><table {...props} /></div>,
};

export function Citation({message, number, onSource}) {
  const source = sourceAt(message, number);
  return source ? <button className="cite" onClick={() => onSource({source, number, messageId: message.id})}
    aria-label={`Источник ${number}: ${source.name}`}>[{number}]</button>
    : <span className="missing-cite" title="Снимок источника не сохранён в этом старом сообщении">[{number}]</span>;
}

export default function MessageMarkdown({message, onSource, text}) {
  return <SourceContext.Provider value={{message, onSource}}><div className="markdown">
    <Markdown skipHtml remarkPlugins={[remarkGfm, citationPlugin]} components={components}>{text ?? message.content}</Markdown>
  </div></SourceContext.Provider>;
}
