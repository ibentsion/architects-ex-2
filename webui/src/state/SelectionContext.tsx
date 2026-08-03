import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import type { SupportCitation, SupportPair } from "../types";

/** What the citation sidebar is currently showing.
 *
 * Both views share it, which is what lets the sidebar be one component instead
 * of two. React Context is enough here — the state is three fields and one
 * consumer tree; a store library would be a dependency with nothing to do.
 */
interface Selection {
  pair: SupportPair | null;
  citation: SupportCitation | null;
  select: (pair: SupportPair | null) => void;
  selectCitation: (citation: SupportCitation | null) => void;
}

const SelectionContext = createContext<Selection | null>(null);

export function SelectionProvider({ children }: { children: ReactNode }) {
  const [pair, setPair] = useState<SupportPair | null>(null);
  const [citation, setCitation] = useState<SupportCitation | null>(null);

  const select = useCallback((next: SupportPair | null) => {
    setPair(next);
    setCitation(null); // a new pair invalidates the open citation
  }, []);

  const value = useMemo<Selection>(
    () => ({ pair, citation, select, selectCitation: setCitation }),
    [pair, citation, select],
  );

  return (
    <SelectionContext.Provider value={value}>{children}</SelectionContext.Provider>
  );
}

export function useSelection(): Selection {
  const value = useContext(SelectionContext);
  if (!value) throw new Error("useSelection must be used inside <SelectionProvider>");
  return value;
}
