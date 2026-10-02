const { test, expect } = require("@playwright/test");

// Charter: a reader opens the real WASM playground, edits code, scrolls to
// earlier/later and blank lines, then clicks and types at visible characters.
// Text, line numbers and the active row must agree at desktop/mobile sizes
// and scaled layout. Stop after the edited character appears on that row.
// Build the module first with ./scripts/build_site_wasm.sh from the repo root.

async function openEditor(page) {
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeEnabled();
  await expect(page.locator("#highlight .line").first()).toBeAttached();
  return page.getByRole("textbox", { name: "Muninn program" });
}

async function alignment(page, scroll) {
  return page.evaluate(async (scroll) => {
    const source = document.querySelector("#source");
    const highlight = document.querySelector("#highlight");
    const gutter = document.querySelector("#gutter");
    source.scrollTop = scroll === "end" ? source.scrollHeight : scroll;
    await new Promise(requestAnimationFrame);
    await new Promise(requestAnimationFrame);
    const rows = [...highlight.children];
    const numbers = [...gutter.children];
    return {
      sourceScroll: source.scrollTop,
      highlightScroll: highlight.scrollTop,
      gutterScroll: gutter.scrollTop,
      viewportDifference: Math.abs(source.clientHeight - gutter.clientHeight),
      rowDifference: Math.max(...rows.map((row, i) =>
        Math.abs(row.getBoundingClientRect().top - numbers[i].getBoundingClientRect().top))),
    };
  }, scroll);
}

for (const scenario of [
  { name: "desktop", width: 1440, height: 900, zoom: 1 },
  { name: "narrow", width: 390, height: 844, zoom: 1 },
  { name: "scaled", width: 1440, height: 900, zoom: 1.25 },
  { name: "zoomed out", width: 1440, height: 900, zoom: 0.75 },
  { name: "slightly zoomed out", width: 1440, height: 900, zoom: 0.85 },
]) {
  test(`${scenario.name}: scrolled code and line numbers stay aligned`, async ({ page }) => {
    await page.setViewportSize({ width: scenario.width, height: scenario.height });
    const editor = await openEditor(page);
    await page.evaluate(zoom => { document.body.style.zoom = zoom; }, scenario.zoom);
    const lines = Array.from({ length: 100 }, (_, i) =>
      i % 4 === 2 ? "" : `let v${i}: Int = ${i};`);
    await editor.fill(lines.join("\n") + "\n");
    await expect(page.locator("#highlight .line")).toHaveCount(101);

    for (const scroll of [0, 173, "end", 0]) {
      const initial = await alignment(page, scroll);
      await expect.poll(async () => {
        const m = await alignment(page, scroll);
        return Math.max(m.rowDifference, m.viewportDifference);
      }, { message: JSON.stringify({ scroll, ...initial }) }).toBeLessThan(1);
      const m = await alignment(page, scroll);
      // At fractional scales WebKit quantizes textarea.scrollTop differently
      // from div.scrollTop (172 versus 171 at 85%). Visible row positions,
      // checked above, and native click positions are the alignment oracle.
      expect(Math.abs(m.sourceScroll - m.highlightScroll)).toBeLessThanOrEqual(1);
      expect(Math.abs(m.sourceScroll - m.gutterScroll)).toBeLessThanOrEqual(1);
      if (scroll !== 0) expect(m.sourceScroll).toBeGreaterThan(0);
    }
  });

  test(`${scenario.name}: clicking visible code inserts at that character`, async ({ page }) => {
    await page.setViewportSize({ width: scenario.width, height: scenario.height });
    const editor = await openEditor(page);
    await page.evaluate(zoom => { document.body.style.zoom = zoom; }, scenario.zoom);
    const lines = Array.from({ length: 80 }, (_, i) =>
      i % 4 === 2 ? "" : `let v${i}: Int = ${i};`);
    const source = lines.join("\n") + "\n";
    await editor.fill(source);
    await expect(page.locator("#highlight .line")).toHaveCount(81);
    await editor.scrollIntoViewIfNeeded();

    // The drawn character supplies click coordinates, but the browser's real
    // textarea determines selectionStart. This catches text/caret drift, not
    // just two CSS layers agreeing with each other.
    for (const index of [0, 17, 78, 79, 80]) {
      const position = await page.evaluate(async (index) => {
        const source = document.querySelector("#source");
        const row = document.querySelector("#highlight").children[index];
        const sourceBounds = source.getBoundingClientRect();
        const delta = row.getBoundingClientRect().top - sourceBounds.top;
        const zoom = Number(getComputedStyle(document.body).zoom);
        source.scrollTop += (delta - sourceBounds.height / 2) / zoom;
        await new Promise(requestAnimationFrame);
        await new Promise(requestAnimationFrame);
        const bounds = row.getBoundingClientRect();
        const walker = document.createTreeWalker(row, NodeFilter.SHOW_TEXT);
        const text = walker.nextNode();
        let x = bounds.left;
        if (text && text.length) {
          const range = document.createRange();
          range.setStart(text, 0);
          range.setEnd(text, 1);
          x = range.getBoundingClientRect().left + 0.1;
        }
        return { x, y: bounds.top + bounds.height / 2 };
      }, index);
      await page.mouse.click(position.x, position.y);
      const expected = lines.slice(0, index).reduce((sum, line) => sum + line.length + 1, 0);
      await expect.poll(() => editor.evaluate(e => e.selectionStart)).toBe(expected);
      await expect(page.locator("#gutter .marked")).toHaveText(String(index + 1));
      await expect(page.locator("#highlight .current")).toHaveCount(1);
      expect(await page.locator("#highlight .current").evaluate(e =>
        [...e.parentNode.children].indexOf(e))).toBe(index);
    }
    await page.keyboard.type("// here");
    await expect(editor).toHaveValue(source + "// here");
    await expect(page.locator("#highlight .current")).toHaveText("// here");
  });

  test(`${scenario.name}: horizontal scrolling keeps long-line character positions`, async ({ page }) => {
    await page.setViewportSize({ width: scenario.width, height: scenario.height });
    const editor = await openEditor(page);
    await page.evaluate(zoom => { document.body.style.zoom = zoom; }, scenario.zoom);
    const longLine = `\tlet message: String = "${"abcd".repeat(80)}";`;
    const source = `// A long, tab-indented line.\n\n${longLine}\nprint(message);\n`;
    await editor.fill(source);
    await expect(page.locator("#highlight .line")).toHaveCount(5);
    await expect(page.locator("#highlight .line").nth(2)).toHaveText(longLine);
    await editor.scrollIntoViewIfNeeded();

    const column = 250;
    const position = await page.evaluate(async column => {
      const source = document.querySelector("#source");
      const row = document.querySelector("#highlight").children[2];
      const walker = document.createTreeWalker(row, NodeFilter.SHOW_TEXT);
      let text = walker.nextNode(), offset = column;
      while (text && offset >= text.length) {
        offset -= text.length;
        text = walker.nextNode();
      }
      const range = document.createRange();
      range.setStart(text, offset);
      range.setEnd(text, offset + 1);
      const zoom = Number(getComputedStyle(document.body).zoom);
      const bounds = source.getBoundingClientRect();
      source.scrollLeft += (range.getBoundingClientRect().left - bounds.left - bounds.width / 2) / zoom;
      await new Promise(requestAnimationFrame);
      await new Promise(requestAnimationFrame);
      const character = range.getBoundingClientRect();
      return { x: character.left + 0.1, y: character.top + character.height / 2 };
    }, column);
    expect(await editor.evaluate(e => e.scrollLeft)).toBeGreaterThan(0);
    await page.mouse.click(position.x, position.y);
    const expected = source.indexOf(longLine) + column;
    await expect.poll(() => editor.evaluate(e => e.selectionStart)).toBe(expected);
    await expect(page.locator("#gutter .marked")).toHaveText("3");
    await page.keyboard.type("X");
    await expect(editor).toHaveValue(source.slice(0, expected) + "X" + source.slice(expected));
    await expect(page.locator("#highlight .current")).toHaveText(
      longLine.slice(0, column) + "X" + longLine.slice(column));
  });
}
