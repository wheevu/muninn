const { test, expect } = require("@playwright/test");
const { spawn } = require("node:child_process");

// Native Safari rounds text line boxes differently from Playwright's WebKit
// build. This regression therefore uses safaridriver, not an engine substitute.
// Requires macOS with Safari's Allow Remote Automation already enabled.
test("native Safari: text, caret and gutter agree at normal and reduced scales", async ({}, testInfo) => {
  test.skip(process.platform !== "darwin", "Native Safari requires macOS.");
  test.setTimeout(60000);
  const driver = spawn("safaridriver", ["--port", "4447"], { stdio: "ignore" });
  console.log(`Owned safaridriver PID ${driver.pid}, port 4447; cleanup in finally.`);
  let session;
  const call = async (path, data, method = data === undefined ? "GET" : "POST") => {
    const response = await fetch(`http://127.0.0.1:4447${path}`, {
      method,
      headers: { "Content-Type": "application/json" },
      body: data === undefined ? undefined : JSON.stringify(data),
      signal: AbortSignal.timeout(15000),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(JSON.stringify(result.value));
    return result.value;
  };
  const execute = async (fn, ...args) => call(`/session/${session}/execute/async`, {
    script: `const done=arguments[arguments.length-1];Promise.resolve((${fn.toString()})(...Array.from(arguments).slice(0,-1))).then(done,e=>done({error:String(e)}));`,
    args,
  });
  try {
    await expect.poll(async () => {
      try { return (await call("/status")).ready; } catch { return false; }
    }, { timeout: 10000 }).toBe(true);
    session = (await call("/session", { capabilities: { alwaysMatch: { browserName: "safari" } } })).sessionId;
    await call(`/session/${session}/window/rect`, { width: 1200, height: 900 });
    await call(`/session/${session}/url`, { url: `${testInfo.project.use.baseURL}/?native-safari-regression=1` });
    await expect.poll(() => execute(() => document.querySelector("#highlight")?.children.length || 0)).toBeGreaterThan(0);

    for (const zoom of [1, 0.85, 0.75]) {
      const metrics = await execute(async zoom => {
        document.body.style.zoom = zoom;
        const source = document.querySelector("#source");
        source.scrollTop = 0;
        source.scrollIntoView({ block: "center", behavior: "instant" });
        await new Promise(requestAnimationFrame);
        await new Promise(requestAnimationFrame);
        const rows = [...document.querySelector("#highlight").children];
        const numbers = [...document.querySelector("#gutter").children];
        const target = rows[6];
        const walker = document.createTreeWalker(target, NodeFilter.SHOW_TEXT);
        const text = walker.nextNode();
        const range = document.createRange();
        range.setStart(text, 0);
        range.setEnd(text, 1);
        const character = range.getBoundingClientRect();
        return {
          difference: Math.max(...rows.map((row, i) => Math.abs(
            row.getBoundingClientRect().top - numbers[i].getBoundingClientRect().top))),
          x: character.left + 0.1,
          y: character.top + character.height / 2,
          expected: source.value.split("\n").slice(0, 6).reduce((sum, line) => sum + line.length + 1, 0),
        };
      }, zoom);
      expect(metrics.error).toBeUndefined();
      expect(metrics.difference, `Row drift at scale ${zoom}`).toBeLessThan(1);
      await call(`/session/${session}/actions`, { actions: [{ type: "pointer", id: "mouse", parameters: { pointerType: "mouse" }, actions: [
        { type: "pointerMove", origin: "viewport", x: Math.round(metrics.x), y: Math.round(metrics.y) },
        { type: "pointerDown", button: 0 }, { type: "pointerUp", button: 0 },
      ] }] });
      expect(await execute(() => document.querySelector("#source").selectionStart)).toBe(metrics.expected);
      expect(await execute(() => document.querySelector("#gutter .marked")?.textContent)).toBe("7");
    }
  } catch (error) {
    if (session) {
      const screenshot = await call(`/session/${session}/screenshot`).catch(() => null);
      if (screenshot) await testInfo.attach("native-safari-failure", {
        body: Buffer.from(screenshot, "base64"), contentType: "image/png",
      });
    }
    throw error;
  } finally {
    if (session) await call(`/session/${session}`, undefined, "DELETE").catch(() => {});
    driver.kill();
  }
});
