import { describe, expect, it } from "vitest";
import { closeOpenMarkdown, renderStreamingMarkdown } from "./markdown";

describe("markdown while it streams", () => {
  it("closes a code block the stream has only opened", () => {
    expect(closeOpenMarkdown("Run this:\n```python\nprint(1)")).toBe("Run this:\n```python\nprint(1)\n```");
    expect(closeOpenMarkdown("```\na\n```\ndone")).toBe("```\na\n```\ndone");
  });

  it("holds back a half-typed bold or code marker at the end", () => {
    expect(closeOpenMarkdown("Use **reb")).toBe("Use reb");
    expect(closeOpenMarkdown("Use **rebase** and `git pu")).toBe("Use **rebase** and git pu");
    expect(closeOpenMarkdown("Use **rebase**.")).toBe("Use **rebase**.");
  });

  it("renders a half-written code block as code, not as text that swallows the rest", () => {
    const html = renderStreamingMarkdown("Two ways:\n- **merge** keeps history\n```bash\ngit rebase main");
    expect(html).toContain("<li>");
    expect(html).toContain("<strong>merge</strong>");
    expect(html).toContain("hlx-code-wrapper");
    expect(html).not.toContain("```");
  });

  it("renders each growing prefix of a reply without a literal marker", () => {
    const reply = "Merge **keeps** history.\n\n```js\nconst a = 1;\n```\nDone with `rebase`.";
    for (let end = 1; end <= reply.length; end += 1) {
      const html = renderStreamingMarkdown(reply.slice(0, end));
      expect(html).not.toContain("```");
      expect(html).not.toMatch(/\*\*/);
    }
  });
});
