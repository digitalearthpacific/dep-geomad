"""Build an HTML file with an old/new before-after slider per region.

Images are referenced as relative paths (tests/output/*.png), not embedded,
so this must stay next to the tests/output/ directory.

Usage:
    python tests/build_compare_html.py
"""

from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
OUT_HTML = Path(__file__).parent / "compare_slider.html"

STYLE = """
body { font-family: -apple-system, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }
h1 { font-size: 1.4rem; }
h2 { margin-top: 3rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }

.slider-container { position: relative; max-width: 100%; line-height: 0; user-select: none; cursor: ew-resize; touch-action: none; }
.slider-container img { display: block; width: 100%; -webkit-user-drag: none; user-select: none; pointer-events: none; }
.img-old-wrap { position: absolute; top: 0; left: 0; height: 100%; overflow: hidden; }
.img-old-wrap img { width: var(--full-width); max-width: none; }

.divider { position: absolute; top: 0; bottom: 0; width: 2px; background: white;
           box-shadow: 0 0 4px rgba(0,0,0,0.6); pointer-events: none; }
.handle { position: absolute; top: 50%; width: 36px; height: 36px; border-radius: 50%;
          background: white; box-shadow: 0 0 6px rgba(0,0,0,0.6);
          transform: translate(-50%, -50%); display: flex; align-items: center; justify-content: center;
          font-size: 1rem; pointer-events: none; }

.label { position: absolute; top: 8px; background: rgba(0,0,0,0.6); color: white;
         font-size: 0.75rem; padding: 2px 6px; border-radius: 3px; pointer-events: none; }
.label-old { left: 8px; }
.label-new { right: 8px; }
"""

SCRIPT = """
document.querySelectorAll('.slider-container').forEach(function (container) {
  var wrap = container.querySelector('.img-old-wrap');
  var divider = container.querySelector('.divider');
  var handle = container.querySelector('.handle');
  var dragging = false;

  function setPct(pct) {
    pct = Math.max(0, Math.min(100, pct));
    wrap.style.width = pct + '%';
    divider.style.left = pct + '%';
    handle.style.left = pct + '%';
  }

  function pctFromEvent(e) {
    var rect = container.getBoundingClientRect();
    var x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    return (x / rect.width) * 100;
  }

  container.addEventListener('pointerdown', function (e) {
    e.preventDefault();
    dragging = true;
    setPct(pctFromEvent(e));
  });
  container.addEventListener('dragstart', function (e) { e.preventDefault(); });
  window.addEventListener('pointermove', function (e) {
    if (dragging) setPct(pctFromEvent(e));
  });
  window.addEventListener('pointerup', function () { dragging = false; });

  // Match the "old" image's rendered width to the container so it lines up pixel-for-pixel.
  function syncWidth() {
    container.style.setProperty('--full-width', container.clientWidth + 'px');
  }
  new ResizeObserver(syncWidth).observe(container);
  syncWidth();

  setPct(50);
});
"""


def main() -> None:
    old_files = sorted(OUTPUT_DIR.glob("*_old.png"))
    sections = []
    for old_path in old_files:
        name = old_path.stem[: -len("_old")]
        new_path = OUTPUT_DIR / f"{name}_new.png"
        if not new_path.exists():
            continue
        title = name.replace("_", " ").title()
        old_rel = f"output/{old_path.name}"
        new_rel = f"output/{new_path.name}"
        sections.append(f"""
<h2>{title}</h2>
<div class="slider-container">
  <img class="img-new" src="{new_rel}" draggable="false">
  <div class="img-old-wrap">
    <img class="img-old" src="{old_rel}" draggable="false">
  </div>
  <div class="divider"></div>
  <div class="handle">&#8596;</div>
  <span class="label label-old">OLD</span>
  <span class="label label-new">NEW</span>
</div>
""")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GeoMAD cloud mask comparison</title>
<style>{STYLE}</style>
</head>
<body>
<h1>Cloud mask comparison: old vs new</h1>
<p>Drag the handle on each image to compare the production ("old") cloud mask
against the candidate ("new") config. Left = old, right = new.</p>
{''.join(sections)}
<script>{SCRIPT}</script>
</body>
</html>
"""
    OUT_HTML.write_text(html)
    print(f"wrote {OUT_HTML} ({len(sections)} regions)")


if __name__ == "__main__":
    main()
