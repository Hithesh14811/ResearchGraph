import { useEffect, useRef, useState } from "react";

import { api } from "../services/api";
import type { WorkflowEvent } from "../types/api";

const TERMINAL = new Set(["completed", "failed", "cancelled", "rejected"]);

/**
 * Subscribes to the run's Server-Sent Events stream. The server replays persisted history
 * first and then follows live events; EventSource reconnects automatically and resumes
 * from the last received event id (Last-Event-ID).
 *
 * Events arrive as default (unnamed) SSE messages with the workflow event type in the JSON
 * payload, so they can never be confused with EventSource's own `error` event.
 */
export function useEventStream(researchId: string | undefined) {
  const [events, setEvents] = useState<WorkflowEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const seen = useRef(new Set<number>());

  useEffect(() => {
    if (!researchId) return;
    seen.current = new Set();
    setEvents([]);
    const source = new EventSource(api.streamUrl(researchId));
    let buffer: WorkflowEvent[] = [];
    let frame = 0;

    const flush = () => {
      frame = 0;
      if (buffer.length === 0) return;
      const batch = buffer;
      buffer = [];
      setEvents((previous) => [...previous, ...batch]);
    };

    source.onmessage = (message: MessageEvent<string>) => {
      let event: WorkflowEvent;
      try {
        event = JSON.parse(message.data) as WorkflowEvent;
      } catch {
        return; // ignore malformed frames rather than crashing the page
      }
      if (seen.current.has(event.seq)) return;
      seen.current.add(event.seq);
      buffer.push(event);
      if (!frame) frame = window.requestAnimationFrame(flush);
      if (event.type === "run_status" && TERMINAL.has(String(event.data.status))) {
        source.close();
        setConnected(false);
      }
    };
    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(source.readyState === EventSource.OPEN);

    return () => {
      source.close();
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [researchId]);

  return { events, connected };
}
