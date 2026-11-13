import { render, screen } from "@testing-library/svelte";
import { tick } from "svelte";
import { describe, expect, it } from "vitest";
import LazyPanel from "./LazyPanel.svelte";
import OptionalParts from "./OptionalParts.svelte";

async function settle() {
  for (let i = 0; i < 5; i++) {
    await new Promise((resolve) => setTimeout(resolve, 0));
    await tick();
  }
}

describe("a Settings panel that loads late", () => {
  it("shows the panel once its chunk has loaded, and nothing before (no error, no empty state of its own)", async () => {
    let finish: (module: { default: typeof OptionalParts }) => void = () => {};
    const loading = new Promise<{ default: typeof OptionalParts }>((resolve) => (finish = resolve));
    const { container } = render(LazyPanel, { props: { load: () => loading } });
    await settle();
    expect(container.querySelector(".lazy-anchor")).not.toBeNull();
    expect(container.textContent?.trim()).toBe("");
    finish({ default: OptionalParts });
    await settle();
    expect(container.querySelector(".lazy-anchor")).toBeNull();
  });

  it("says so in one line when its chunk cannot be loaded", async () => {
    render(LazyPanel, { props: { load: () => Promise.reject(new Error("chunk missing")) } });
    await settle();
    expect(
      screen.getByText("This part of Settings could not be loaded. Close Settings and open it again."),
    ).toBeTruthy();
  });
});
