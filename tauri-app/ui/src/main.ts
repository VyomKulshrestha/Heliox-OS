import "@fontsource-variable/space-grotesk/wght.css";
import "@fontsource-variable/jetbrains-mono/wght.css";
import "./i18n";
import App from "./App.svelte";
import { mount } from "svelte";
import { installWaylandShortcut } from "./lib/waylandShortcut";
import { installLinkGuard } from "./lib/utils/links";
import { installErrorRecorder } from "./lib/errorRecorder";

// First in the page's own code (imports run before it), so it sees the app mount.
installErrorRecorder();

const savedTheme = localStorage.getItem("theme");
if (savedTheme === "light" || (!savedTheme && window.matchMedia("(prefers-color-scheme: light)").matches)) {
  document.documentElement.classList.add("light-mode");
} else {
  document.documentElement.classList.remove("light-mode");
}

installWaylandShortcut();
installLinkGuard(); // a link never takes the window away from the app

const app = mount(App, {
  target: document.getElementById("app")!,
});

export default app;
