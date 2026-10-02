const { defineConfig } = require("@playwright/test");

module.exports = defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  forbidOnly: true,
  retries: 0,
  workers: 2,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:8778",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "chrome", testIgnore: "safari.spec.cjs", use: { browserName: "chromium", channel: "chrome", deviceScaleFactor: 2 } },
    { name: "webkit", testIgnore: "safari.spec.cjs", use: { browserName: "webkit", deviceScaleFactor: 2 } },
    { name: "safari", testMatch: "safari.spec.cjs", workers: 1 },
  ],
  webServer: {
    command: "python3 -m http.server 8778 --bind 127.0.0.1",
    cwd: __dirname,
    url: "http://127.0.0.1:8778",
    reuseExistingServer: false,
  },
});
