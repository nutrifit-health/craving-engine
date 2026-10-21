'use client';

import { useMemo, useState } from 'react';
import { Button } from '@heroui/react';
import { SvgIcon } from '@nutrifit/web-ui/SvgIcon';

interface PythonCodeViewerProps {
  code: string;
  filename?: string;
}

/**
 * Высокопроизводительный компонент подсветки синтаксиса Python
 * в стиле VS Code / Prism Dark с номерами строк и копированием.
 */
export function PythonCodeViewer({ code, filename = 'experiment.py' }: PythonCodeViewerProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Разбиение на строки и подсветка ключевых токенов Python
  const highlightedLines = useMemo(() => {
    const lines = code.split('\n');

    return lines.map((line) => {
      // 1. Комментарии (# ...)
      const commentIndex = line.indexOf('#');
      let codePart = line;
      let commentPart = '';

      if (commentIndex !== -1) {
        // Проверяем, не внутри ли кавычек
        const before = line.slice(0, commentIndex);
        const quotesSingle = (before.match(/'/g) || []).length;
        const quotesDouble = (before.match(/"/g) || []).length;
        if (quotesSingle % 2 === 0 && quotesDouble % 2 === 0) {
          codePart = before;
          commentPart = line.slice(commentIndex);
        }
      }

      // 2. Токенизация кода Python
      // Регулярное выражение для строк, ключевых слов, чисел, операторов и идентификаторов
      const tokenRegex = /("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:def|class|import|from|as|return|if|elif|else|while|for|in|try|except|finally|with|yield|lambda|pass|break|continue|raise|assert|async|await|and|or|not|is)\b|\b(?:self|cls|True|False|None)\b|\b(?:int|float|str|bool|list|dict|set|tuple|print|len|range|enumerate|zip|min|max|sum|abs|round|open|super)\b|@[a-zA-Z_]\w*|\b\d+(?:\.\d+)?\b|[a-zA-Z_]\w*|[^\s\w]+|\s+)/g;

      const tokens: { text: string; className: string }[] = [];
      let match: RegExpExecArray | null;

      while ((match = tokenRegex.exec(codePart)) !== null) {
        const text = match[0];
        let className = 'text-zinc-200'; // обычный текст

        if (text.startsWith('"') || text.startsWith("'")) {
          className = 'text-emerald-400 dark:text-emerald-300'; // строки
        } else if (/^(?:def|class|import|from|as|return|if|elif|else|while|for|in|try|except|finally|with|yield|lambda|pass|break|continue|raise|assert|async|await|and|or|not|is)$/.test(text)) {
          className = 'text-purple-400 font-semibold'; // ключевые слова
        } else if (/^(?:self|cls|True|False|None)$/.test(text)) {
          className = 'text-amber-400 font-medium'; // константы и self
        } else if (/^(?:int|float|str|bool|list|dict|set|tuple|print|len|range|enumerate|zip|min|max|sum|abs|round|open|super)$/.test(text)) {
          className = 'text-sky-400'; // встроенные функции и типы
        } else if (text.startsWith('@')) {
          className = 'text-yellow-400 font-medium'; // декораторы
        } else if (/^\d+(?:\.\d+)?$/.test(text)) {
          className = 'text-orange-400'; // числа
        } else if (/^[=+\-*/%&|^~<>!]=?|->$/.test(text)) {
          className = 'text-cyan-400'; // операторы
        }

        tokens.push({ text, className });
      }

      return { tokens, commentPart };
    });
  }, [code]);

  return (
    <div className='rounded-2xl bg-zinc-950 overflow-hidden flex flex-col font-mono text-xs'>
      {/* Code Header Bar */}
      <div className='flex items-center justify-between px-4 py-2.5 bg-zinc-900 select-none'>
        <div className='flex items-center gap-2'>
          <span className='p-1 rounded bg-emerald-500/10 text-emerald-400'>
            <SvgIcon aria-hidden='true' iconName='solar:code-linear' width={14} />
          </span>
          <span className='font-semibold text-zinc-300 font-sans'>{filename}</span>
          <span className='text-[10px] text-zinc-500 bg-zinc-800 px-2 py-0.5 rounded'>Python 3</span>
          <span className='text-[10px] text-zinc-500'>{highlightedLines.length} строк</span>
        </div>

        <Button
          size='sm'
          variant='secondary'
          onPress={handleCopy}
          className='h-7 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-300'
        >
          <SvgIcon aria-hidden='true' iconName={copied ? 'solar:check-circle-bold' : 'solar:copy-linear'} width={13} />
          <span>{copied ? 'Скопировано!' : 'Копировать код'}</span>
        </Button>
      </div>

      {/* Code Content with Line Numbers */}
      <div className='overflow-x-auto max-h-[calc(100vh-280px)] overflow-y-auto p-3 [scrollbar-width:thin]'>
        <table className='w-full border-collapse leading-relaxed'>
          <tbody>
            {highlightedLines.map((lineData, lineIdx) => (
              <tr key={lineIdx} className='hover:bg-zinc-900/50 transition-colors'>
                {/* Line number */}
                <td className='w-12 text-right pr-4 text-zinc-600 select-none text-[11px] align-top py-0.5 border-r border-zinc-800/80'>
                  {lineIdx + 1}
                </td>
                {/* Line content */}
                <td className='pl-4 whitespace-pre font-mono py-0.5 text-xs align-top'>
                  {lineData.tokens.map((token, tIdx) => (
                    <span key={tIdx} className={token.className}>
                      {token.text}
                    </span>
                  ))}
                  {lineData.commentPart && (
                    <span className='text-zinc-500 italic'>
                      {lineData.commentPart}
                    </span>
                  )}
                  {lineData.tokens.length === 0 && !lineData.commentPart && '\n'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
