import React, { useEffect, useState } from 'react';

export default function SqlEditorComponent({ value, onChange }) {
  const [MonacoEditor, setMonacoEditor] = useState(null);
  const [monacoError, setMonacoError] = useState(false);

  useEffect(() => {
    let mounted = true;
    // try dynamic import to avoid build/runtime issues; if fails use textarea fallback
    import('@monaco-editor/react')
      .then(mod => { if (mounted) setMonacoEditor(() => mod.default); })
      .catch(err => { console.warn('Monaco dynamic import failed, falling back to textarea', err); setMonacoError(true); });
    return () => { mounted = false; };
  }, []);

  if (monacoError || !MonacoEditor) {
    // simple textarea fallback that works in all environments
    return (
      <textarea
        value={value}
        onChange={e => onChange && onChange(e.target.value)}
        style={{ width: '100%', height: '100%', border: 'none', padding: 8, fontFamily: 'monospace', fontSize: 13, boxSizing: 'border-box' }}
      />
    );
  }

  const Editor = MonacoEditor;
  return (
    <Editor
      height="100%"
      language="sql"
      theme="vs-light"
      value={value}
      onChange={onChange}
      options={{ minimap: { enabled: false }, fontSize: 14, wordWrap: 'on' }}
    />
  );
}