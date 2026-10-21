'use client';

import { useMemo, useState } from 'react';
import Markdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Button } from '@heroui/react';
import { SvgIcon } from '@nutrifit/web-ui/SvgIcon';
import { PythonCodeViewer } from './PythonCodeViewer';

interface MarkdownViewerProps {
  content: string;
  className?: string;
}

/**
 * Отдельная строка $$…$$ становится блочной формулой KaTeX;
 * примеры внутри блоков кода сохраняются буквально.
 */
function normalizeDisplayMath(value: string): string {
  let fenceCharacter = '';
  let fenceLength = 0;

  return value
    .split('\n')
    .map((line) => {
      const fence = /^ {0,3}(`{3,}|~{3,})/.exec(line);
      if (fence) {
        const marker = fence[1];
        if (!fenceCharacter) {
          fenceCharacter = marker[0];
          fenceLength = marker.length;
        } else if (
          marker[0] === fenceCharacter &&
          marker.length >= fenceLength &&
          line.slice(fence[0].length).trim() === ''
        ) {
          fenceCharacter = '';
          fenceLength = 0;
        }
        return line;
      }
      if (fenceCharacter || /^(?: {4}|\t)/.test(line)) return line;

      const formula = /^ {0,3}\$\$(.+)\$\$\s*$/.exec(line);
      return formula ? `$$\n${formula[1].trim()}\n$$` : line;
    })
    .join('\n');
}

function CodeBlockWithCopy({ language, code }: { language: string; code: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (language === 'python' || language === 'py') {
    return (
      <div className='my-3'>
        <PythonCodeViewer code={code} filename='snippet.py' />
      </div>
    );
  }

  return (
    <div className='my-3 rounded-2xl bg-zinc-950 overflow-hidden text-xs font-mono'>
      <div className='flex items-center justify-between px-3.5 py-2 bg-zinc-900 text-[11px] text-zinc-400'>
        <span className='uppercase font-semibold tracking-wider text-zinc-300'>{language || 'text'}</span>
        <Button
          size='sm'
          variant='ghost'
          onPress={handleCopy}
          className='h-6 px-2 text-[10px] text-zinc-400 hover:text-zinc-200'
        >
          <SvgIcon aria-hidden='true' iconName={copied ? 'solar:check-circle-bold' : 'solar:copy-linear'} width={13} />
          <span>{copied ? 'Скопировано' : 'Копировать'}</span>
        </Button>
      </div>
      <pre className='p-3.5 text-zinc-200 overflow-x-auto whitespace-pre-wrap leading-relaxed'>
        {code}
      </pre>
    </div>
  );
}

export function MarkdownViewer({ content, className = '' }: MarkdownViewerProps) {
  const processedContent = useMemo(() => normalizeDisplayMath(content), [content]);

  const components: Components = useMemo(
    () => ({
      h1: ({ children, ...props }) => (
        <h1 className='text-xl font-bold mt-6 mb-3 text-foreground border-b border-default-200 pb-2 flex items-center gap-2' {...props}>
          {children}
        </h1>
      ),
      h2: ({ children, ...props }) => (
        <h2 className='text-lg font-bold mt-5 mb-2.5 text-foreground flex items-center gap-2' {...props}>
          {children}
        </h2>
      ),
      h3: ({ children, ...props }) => (
        <h3 className='text-base font-semibold mt-4 mb-2 text-foreground' {...props}>
          {children}
        </h3>
      ),
      h4: ({ children, ...props }) => (
        <h4 className='text-sm font-semibold mt-3 mb-1.5 text-foreground' {...props}>
          {children}
        </h4>
      ),
      p: ({ children, ...props }) => (
        <p className='my-2.5 leading-relaxed text-sm text-foreground/90' {...props}>
          {children}
        </p>
      ),
      ul: ({ children, ...props }) => (
        <ul className='list-disc pl-5 my-2.5 space-y-1 text-sm text-foreground/90' {...props}>
          {children}
        </ul>
      ),
      ol: ({ children, ...props }) => (
        <ol className='list-decimal pl-5 my-2.5 space-y-1 text-sm text-foreground/90' {...props}>
          {children}
        </ol>
      ),
      li: ({ children, ...props }) => (
        <li className='leading-relaxed' {...props}>
          {children}
        </li>
      ),
      blockquote: ({ children, ...props }) => (
        <blockquote
          className='border-l-4 border-primary pl-4 py-1.5 my-3 bg-primary/5 rounded-r-xl text-default-600 dark:text-default-400 italic text-sm'
          {...props}
        >
          {children}
        </blockquote>
      ),
      hr: (props) => <hr className='my-5 border-default-200' {...props} />,
      a: ({ href, children, ...props }) => (
        <a
          href={href}
          target='_blank'
          rel='noopener noreferrer'
          className='text-primary font-medium underline underline-offset-2 hover:text-primary-600 transition-colors'
          {...props}
        >
          {children}
        </a>
      ),
      table: ({ children, ...props }) => (
        <div className='overflow-x-auto my-4 rounded-2xl bg-default-100/60'>
          <table className='min-w-full divide-y divide-default-200/60 text-xs' {...props}>
            {children}
          </table>
        </div>
      ),
      thead: ({ children, ...props }) => (
        <thead className='bg-default-100/80 text-foreground font-semibold' {...props}>
          {children}
        </thead>
      ),
      th: ({ children, ...props }) => (
        <th className='px-3.5 py-2.5 text-left font-bold tracking-wider text-default-700' {...props}>
          {children}
        </th>
      ),
      td: ({ children, ...props }) => (
        <td className='px-3.5 py-2.5 border-t border-default-100 text-default-600' {...props}>
          {children}
        </td>
      ),
      code: ({ className: codeClassName, children, ...rest }) => {
        const isBlock = typeof codeClassName === 'string' && codeClassName.startsWith('language-');
        if (isBlock) {
          const language = codeClassName.slice('language-'.length) || 'plaintext';
          const codeString = String(children).replace(/\n$/, '');
          return <CodeBlockWithCopy language={language} code={codeString} />;
        }
        return (
          <code className='px-1.5 py-0.5 rounded bg-default-100 text-primary dark:text-primary-400 text-xs font-mono font-medium' {...rest}>
            {children}
          </code>
        );
      },
      pre: ({ children }) => <>{children}</>,
      span: ({ className: spanClassName, children, ...props }) => {
        return (
          <span
            className={spanClassName}
            tabIndex={spanClassName === 'katex-display' ? 0 : undefined}
            {...props}
          >
            {children}
          </span>
        );
      },
    }),
    []
  );

  return (
    <div className={`prose dark:prose-invert max-w-none text-foreground ${className}`}>
      <Markdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[
          [
            rehypeKatex,
            {
              trust: false,
              strict: 'ignore',
              output: 'htmlAndMathml',
            },
          ],
        ]}
        components={components}
      >
        {processedContent}
      </Markdown>
    </div>
  );
}
