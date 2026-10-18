export async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(`/api${path}`, {
      ...options,
      headers: {...(options.body && !(options.body instanceof FormData) ? {'Content-Type': 'application/json'} : {}), ...options.headers},
    });
  } catch {
    throw new Error('Сервер недоступен. Проверьте, запущен ли CV Screener, и повторите запрос.');
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const error = new Error(typeof body.detail === 'string' ? body.detail : 'Не удалось обработать запрос. Обновите страницу и повторите.');
    error.status = response.status;
    throw error;
  }
  return response.json();
}

export const post = (path, body) => api(path, {method: 'POST', body: JSON.stringify(body)});

export function sourceAt(message, number) {
  return Number.isInteger(number) && number > 0 ? message?.sources?.[number - 1] : undefined;
}

export function citationPlugin() {
  return tree => {
    function walk(node) {
      if (!node.children || ['link', 'code', 'inlineCode'].includes(node.type)) return;
      node.children = node.children.flatMap(child => {
        if (child.type !== 'text') { walk(child); return [child]; }
        const parts = []; let last = 0;
        for (const match of child.value.matchAll(/\[(\d+)\]/g)) {
          if (match.index > last) parts.push({type: 'text', value: child.value.slice(last, match.index)});
          parts.push({type: 'link', url: `#cv-source-${Number(match[1])}`, children: [{type: 'text', value: match[0]}]});
          last = match.index + match[0].length;
        }
        if (last < child.value.length) parts.push({type: 'text', value: child.value.slice(last)});
        return parts;
      });
    }
    walk(tree);
  };
}
