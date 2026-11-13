import { render } from "@testing-library/svelte";
import { tick } from "svelte";
import { describe, expect, it, vi } from "vitest";
import LogMessage from "../../components/LogMessage.svelte";
import { invoke } from "../api/invoke";
import { byThePerson, installLinkGuard, leavesApp, handleLinkClick, webLink } from "./links";
import { renderMarkdown, renderStreamingMarkdown } from "./markdown";

// Text the window did not write (a model's reply, a page, a file, a plugin) trying each way HTML can run or leave.
const PAYLOADS = [
  "<script>window.__ran = 1</script>",
  '<img src="x" onerror="window.__ran = 1">',
  "[click](javascript:window.__ran=1)",
  '<a href="javascript:window.__ran=1">click</a>',
  '<svg><script>window.__ran = 1</script><circle onload="window.__ran=1"/></svg>',
  '<a href="data:text/html,<script>window.__ran=1</script>">data</a>',
  '<meta http-equiv="refresh" content="0;url=https://example.com/">',
  '<iframe src="https://example.com/"></iframe>',
  '<object data="https://example.com/x.swf"></object>',
  '<form action="https://example.com/"><button>go</button></form>',
  '<base href="https://example.com/">',
  '<div style="background:url(javascript:window.__ran=1)" onclick="window.__ran=1">x</div>',
];

function assertInert(container: Element) {
  expect(container.querySelector("script, iframe, object, embed, meta, base, form, frame")).toBeNull();
  for (const element of container.querySelectorAll("*")) {
    for (const attribute of element.getAttributeNames()) {
      expect(attribute.toLowerCase().startsWith("on"), `${element.tagName} ${attribute}`).toBe(false);
      const value = element.getAttribute(attribute) ?? "";
      if (["href", "src", "action", "formaction", "xlink:href", "data"].includes(attribute.toLowerCase())) {
        expect(/^\s*(javascript|data:text|vbscript):/i.test(value), `${attribute}=${value}`).toBe(false);
      }
    }
  }
  expect((window as unknown as { __ran?: number }).__ran).toBeUndefined();
}

describe("text the window did not write stays inert", () => {
  it.each(PAYLOADS)("in chat markdown: %s", (payload) => {
    for (const html of [renderMarkdown(payload), renderStreamingMarkdown(payload)]) {
      const host = document.createElement("div");
      host.innerHTML = html;
      assertInert(host);
    }
  });

  it.each(PAYLOADS)("in the log view (LogMessage): %s", async (payload) => {
    const { container } = render(LogMessage, { props: { content: `**reply**\n\n${payload}`, role: "agent" } });
    await tick();
    await new Promise((resolve) => setTimeout(resolve, 0));
    await tick();
    assertInert(container);
  });
});

describe("a link never takes the window away from the app", () => {
  // A test cannot make a trusted event, so "the person clicked" is passed in; the last two tests use the real check.
  function click(href: string, target = "", byPerson = true): { prevented: boolean; opened: string[] } {
    const anchor = document.createElement("a");
    anchor.setAttribute("href", href);
    if (target) anchor.target = target;
    document.body.append(anchor);
    const opened: string[] = [];
    const event = new MouseEvent("click", { bubbles: true, cancelable: true });
    anchor.addEventListener("click", (e) =>
      handleLinkClick(
        e,
        (url) => opened.push(url),
        () => byPerson,
      ),
    );
    anchor.dispatchEvent(event);
    anchor.remove();
    return { prevented: event.defaultPrevented, opened };
  }

  it("opens nothing for a click a script dispatched, and still keeps the window", () => {
    expect(click("https://example.com/?q=secret", "", false)).toEqual({ prevented: true, opened: [] });
    expect(click("https://example.com/", "_blank", false)).toEqual({ prevented: true, opened: [] });
  });

  it("asks the browser whether the person clicked (isTrusted), through the one guard on the document", () => {
    const opened: string[] = [];
    const host = document.createElement("div");
    host.innerHTML = '<a href="https://example.com/auto">x</a>';
    document.body.append(host);
    const remove = installLinkGuard(document);
    const anchor = host.querySelector("a") as HTMLAnchorElement;
    const event = new MouseEvent("click", { bubbles: true, cancelable: true });
    expect(byThePerson(event)).toBe(false); // dispatched by this test, like any script's click
    anchor.addEventListener("click", (e) => {
      if (!e.defaultPrevented) opened.push("followed");
    });
    anchor.dispatchEvent(event);
    anchor.click(); // what a stray auto-clicking script would do
    remove();
    host.remove();
    expect(event.defaultPrevented).toBe(true);
    expect(opened).toEqual([]);
    expect(vi.mocked(invoke)).not.toHaveBeenCalledWith("open_external_link", expect.anything());
  });

  it("opens a web or mail link outside, in the system's browser", () => {
    expect(click("https://example.com/a?b=c")).toEqual({ prevented: true, opened: ["https://example.com/a?b=c"] });
    expect(click("mailto:sam@example.com").opened).toEqual(["mailto:sam@example.com"]);
    expect(click("https://example.com/", "_blank").prevented).toBe(true);
  });

  it("opens nothing for script, data or file addresses", () => {
    for (const href of ["javascript:window.__ran=1", "data:text/html,<b>x</b>", "file:///etc/passwd"]) {
      expect(click(href), href).toEqual({ prevented: true, opened: [] });
    }
  });

  it("leaves the app's own anchors alone", () => {
    expect(click("#section")).toEqual({ prevented: false, opened: [] });
    const anchor = document.createElement("a");
    anchor.setAttribute("href", "/settings");
    expect(leavesApp(anchor)).toBe(false);
  });

  it("makes only web addresses into plugin links", () => {
    expect(webLink("https://github.com/x/y")).toBe("https://github.com/x/y");
    for (const url of ["javascript:alert(1)", "data:text/html,x", "file:///etc/passwd", "", undefined, "not a url"]) {
      expect(webLink(url)).toBeUndefined();
    }
  });
});

vi.mock("../api/invoke", () => ({ invoke: vi.fn(async () => undefined) }));

describe("the page's error recorder", () => {
  it("keeps that errors happened, never their text, and only so many", async () => {
    const { installErrorRecorder, LIMIT, MAX_TEXT } = await import("../errorRecorder");
    const frame = document.createElement("iframe");
    document.body.append(frame);
    const target = frame.contentWindow as Window;
    const recorded = installErrorRecorder(target);
    const typed = "my password is hunter2";
    target.dispatchEvent(
      new ErrorEvent("error", {
        message: `Unexpected token in JSON: "${typed}"`,
        error: new SyntaxError(typed),
        filename: "tauri://localhost/assets/index-abc.js?x=" + typed,
        lineno: 12,
      }),
    );
    const violation = new Event("securitypolicyviolation") as Event & Record<string, string>;
    Object.assign(violation, { effectiveDirective: "frame-src", blockedURI: `https://example.com/page?q=${typed}` });
    target.document.dispatchEvent(violation);
    expect(recorded).toEqual([
      { kind: "error", message: "SyntaxError at index-abc.js:12" },
      { kind: "policy", message: "frame-src blocked https://example.com" },
    ]);
    expect(JSON.stringify(recorded)).not.toContain("hunter2");
    for (let i = 0; i < LIMIT + 10; i++)
      target.dispatchEvent(new ErrorEvent("error", { error: new Error("x".repeat(500)) }));
    expect(recorded.length).toBe(LIMIT);
    expect(recorded.every((entry) => entry.message.length <= MAX_TEXT)).toBe(true);
    expect((target as unknown as { __helioxErrors: unknown }).__helioxErrors).toBe(recorded);
    frame.remove();
  });
});
