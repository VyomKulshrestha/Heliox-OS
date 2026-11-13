import { invoke } from "../api/invoke";

/**
 * A link in text the window shows (a model's reply, a page's text, a file's contents, a plugin's description) never
 * takes the window to another page: a page Heliox does not control would then look like Heliox. A click on any link
 * that leaves the app is stopped here. Only a click the person made (the browser marks it trusted; Enter on a focused
 * link and a middle click are the person's too) opens a web or mail address in the system's browser or mail app, and
 * the Rust command checks the scheme again; anything else (javascript:, data:, file:) goes nowhere. A click a script
 * dispatched opens nothing: the browser opening an address the person never clicked is what a stray piece of
 * rendered markup could otherwise cause. This is the only place in the page that opens an address. The Rust side
 * refuses any navigation away from the app's own pages and opens nothing for it (src-tauri/src/navigation.rs).
 */
export function leavesApp(anchor: HTMLAnchorElement, here: Location = window.location): boolean {
  const href = anchor.getAttribute("href");
  if (href === null || href === "" || href.startsWith("#")) return false;
  let target: URL;
  try {
    target = new URL(href, here.href);
  } catch {
    return true;
  }
  return target.origin !== here.origin || anchor.target !== "" || anchor.hasAttribute("download");
}

export function opensOutside(url: string): boolean {
  try {
    return ["http:", "https:", "mailto:"].includes(new URL(url).protocol);
  } catch {
    return false;
  }
}

/** Whether the person made this click (the browser sets isTrusted only for real input); tests pass their own. */
export function byThePerson(event: Event): boolean {
  return event.isTrusted;
}

export function handleLinkClick(
  event: MouseEvent,
  open: (url: string) => unknown = openOutside,
  trusted: (event: Event) => boolean = byThePerson,
): void {
  const element = event.target instanceof Element ? event.target.closest("a") : null;
  if (!(element instanceof HTMLAnchorElement) || !leavesApp(element)) return;
  event.preventDefault();
  event.stopPropagation();
  if (!trusted(event)) return;
  const url = new URL(element.getAttribute("href") ?? "", window.location.href).href;
  if (opensOutside(url)) open(url);
}

/** A plugin's or the marketplace's address as a link only when it is a web address; anything else (javascript:,
 * data:, file:) becomes no link at all, so it cannot run or open anything even before the click guard. */
export function webLink(url: string | null | undefined): string | undefined {
  if (!url) return undefined;
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:" || parsed.protocol === "http:" ? parsed.href : undefined;
  } catch {
    return undefined;
  }
}

function openOutside(url: string): void {
  invoke("open_external_link", { url }).catch((error: unknown) => {
    console.warn("The link could not be opened:", error);
  });
}

/** Stop links leaving the app, for the whole document (clicks and middle clicks). Returns the remover. */
export function installLinkGuard(target: Document = document): () => void {
  const onClick = (event: Event) => handleLinkClick(event as MouseEvent);
  target.addEventListener("click", onClick, true);
  target.addEventListener("auxclick", onClick, true);
  return () => {
    target.removeEventListener("click", onClick, true);
    target.removeEventListener("auxclick", onClick, true);
  };
}
