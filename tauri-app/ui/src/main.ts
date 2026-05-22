import App from "./App.svelte";
import { mount } from "svelte";

const savedTheme = localStorage.getItem("theme");
if (savedTheme === "light" || (!savedTheme && window.matchMedia("(prefers-color-scheme: light)").matches)) {
  document.documentElement.classList.add("light-mode");
} else {
  document.documentElement.classList.remove("light-mode");
}

const app = mount(App, {
  target: document.getElementById("app")!,
});

export default app;