import { useEffect, useState } from "react";

import { api } from "../services/api";
import type { Evidence, Finding, PlanResponse, Report, SourceItem } from "../types/api";

export interface Artifacts {
  report: Report | null;
  sources: SourceItem[];
  evidence: Evidence[];
  findings: Finding[];
  plan: PlanResponse | null;
}

const EMPTY: Artifacts = { report: null, sources: [], evidence: [], findings: [], plan: null };

/** Loads research artifacts; re-fetches whenever ``version`` changes. */
export function useArtifacts(researchId: string | undefined, version: string, includeReport: boolean) {
  const [artifacts, setArtifacts] = useState<Artifacts>(EMPTY);

  useEffect(() => {
    if (!researchId) return;
    let cancelled = false;
    void (async () => {
      const [sources, evidence, findings, plan, report] = await Promise.all([
        api.getSources(researchId).catch(() => ({ items: [] as SourceItem[] })),
        api.getEvidence(researchId).catch(() => ({ items: [] as Evidence[] })),
        api.getFindings(researchId).catch(() => ({ items: [] as Finding[] })),
        api.getPlan(researchId).catch(() => null),
        includeReport ? api.getReport(researchId).catch(() => null) : Promise.resolve(null),
      ]);
      if (!cancelled) {
        setArtifacts({ sources: sources.items, evidence: evidence.items, findings: findings.items, plan, report });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [researchId, version, includeReport]);

  return artifacts;
}
