/** Per-message text direction.
 *
 * The app is RTL, but answers quote policy names, model ids and numbers in
 * Latin script, and a Latin-first message inside an RTL bubble renders with
 * its punctuation in the wrong place. So each bubble gets its own `dir`,
 * decided by the first strong-directional character — the same rule the
 * Unicode bidi algorithm uses for a paragraph.
 */
export type Direction = "rtl" | "ltr";

const RTL = /[֐-޿יִ-﷿ﹰ-﻿]/; // Hebrew, Arabic + forms
const LTR = /[A-Za-zÀ-ʯ]/;

export function detectDir(text: string | null | undefined): Direction {
  if (!text) return "rtl";
  for (const char of text) {
    if (RTL.test(char)) return "rtl";
    if (LTR.test(char)) return "ltr";
  }
  return "rtl"; // digits/punctuation only — this is a Hebrew app
}
