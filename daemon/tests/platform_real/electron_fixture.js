// An Electron window showing field_fixture.html (a text area and a password input), for apps_fields_check.py.
// The page is given in HELIOX_FIXTURE_PAGE. Run with: electron electron_fixture.js [--force-renderer-accessibility]
const { app, BrowserWindow } = require("electron");

app.whenReady().then(() => {
  const window = new BrowserWindow({ width: 640, height: 320, title: "Heliox field fixture" });
  window.loadFile(process.env.HELIOX_FIXTURE_PAGE);
  console.log("ready");
});

app.on("window-all-closed", () => app.quit());
