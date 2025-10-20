import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Ensure React Refresh globals exist (stub) to avoid runtime error
// thrown by @vitejs/plugin-react when preamble isn't detected in some setups.
if (typeof window !== 'undefined') {
  if (!window.$RefreshReg$) window.$RefreshReg$ = () => {};
  if (!window.$RefreshSig$) window.$RefreshSig$ = () => () => {};
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
