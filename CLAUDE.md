# Project conventions

## Figures (ALWAYS)
- **Every figure uses a base font size of 18** for all text: titles, axis labels,
  tick labels, legends, and annotations. 18 is the floor, never smaller.
- Enforce it by importing the shared style at the top of any plotting script:
  `import _style` (see `rxnfp_clustering/scripts/_style.py`) — it sets matplotlib
  rcParams to 18pt and dpi 300. When you add an inline `fontsize=` argument it
  must be `>= 18` (it would otherwise override the rcParams default).
- Save at `dpi=300` with `bbox_inches="tight"`.
- No text overlap: enlarge the figure and move dense legends outside the axes
  rather than shrinking text below 18. No gray explanatory footnote captions
  baked into the figure — context belongs in the slide/prose.

## Beamer deck (ALWAYS)
- The `rxnfp_clustering/slides` deck MUST use the **seahorse** beamer colortheme
  (`\usetheme{default}` + `\usecolortheme{seahorse}`) — the soft, light look the
  user prefers. Do NOT replace it with a dark/navy custom scheme. Chrome
  (frametitle bars, block titles, Part dividers, takeaway boxes) is tinted from
  seahorse's `structure` colour, kept light. The family-colour palette
  (`cStruct`/`cPhys`/… = figures' Okabe-Ito `FAM_COLOR`) is content colour and
  stays as-is; only the *chrome* follows seahorse.
