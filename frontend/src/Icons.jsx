import React from 'react';
const paths = {
  mark: <path d="M4 5h16v4H4zM4 13h9v3H4zM4 20h16" />,
  plus: <path d="M12 5v14M5 12h14" />,
  folder: <path d="M3 7h7l2 2h9v11H3zM3 7V4h7l2 3" />,
  star: <path d="m12 3 2.8 5.7 6.2.9-4.5 4.4 1.1 6.2-5.6-3-5.6 3 1.1-6.2L3 9.6l6.2-.9z" />,
  file: <path d="M14 3H5v18h14V8zM14 3v5h5M8 12h8M8 16h6" />,
  search: <><circle cx="10.5" cy="10.5" r="6.5" /><path d="m16 16 5 5" /></>,
  check: <path d="m5 12 4 4L19 6" />,
  arrow: <path d="M4 12h16m-6-6 6 6-6 6" />,
  close: <path d="m6 6 12 12M6 18 18 6" />,
  download: <path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5" />,
  upload: <path d="M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5" />,
  compare: <path d="M4 5h6v14H4zM14 5h6v14h-6zM7 9v6M17 9v6" />,
  info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6M12 7h.01" /></>,
  dash: <path d="M5 12h14" />,
};
export default function Icon({name}) {
  return <svg viewBox="0 0 24 24" aria-hidden="true">{paths[name] || paths.file}</svg>;
}
