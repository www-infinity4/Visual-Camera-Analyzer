/**
 * useElementStream.js — React hook for the SSE element stream
 *
 * Connects to GET /stream/elements (Server-Sent Events) and returns
 * the latest element readings, connection status, and last timestamp.
 *
 * The hook auto-reconnects on error with a 3-second back-off.
 *
 * @returns {{ elements: Object, status: string, lastTs: number|null }}
 */
import { useEffect, useRef, useState } from 'react';

const SSE_URL = '/stream/elements';
const RECONNECT_DELAY_MS = 3000;

export function useElementStream() {
  const [elements, setElements] = useState({});
  const [status,   setStatus]   = useState('connecting');
  const [lastTs,   setLastTs]   = useState(null);
  const esRef = useRef(null);

  useEffect(() => {
    let reconnectTimer = null;

    function connect() {
      if (esRef.current) esRef.current.close();

      const es = new EventSource(SSE_URL);
      esRef.current = es;

      es.onopen = () => setStatus('live');

      es.onmessage = (e) => {
        try {
          const payload = JSON.parse(e.data);
          setElements(payload.elements || {});
          setLastTs(payload.timestamp || Date.now() / 1000);
          setStatus('live');
        } catch (_) { /* malformed frame — ignore */ }
      };

      es.onerror = () => {
        setStatus('disconnected');
        es.close();
        reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
      };
    }

    connect();

    return () => {
      if (esRef.current) esRef.current.close();
      if (reconnectTimer)  clearTimeout(reconnectTimer);
    };
  }, []);

  return { elements, status, lastTs };
}
