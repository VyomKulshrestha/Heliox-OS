import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { protectedPackageExample } from "../firstRun";

describe("Setup wizard protected packages", () => {
  const source = readFileSync(resolve("src/lib/components/SetupWizard.svelte"), "utf8");

  it("starts with nothing protected on every system", () => {
    // It started as "firefox, nautilus" everywhere, so a Windows first run saved two Linux names into
    // restrictions.toml, and every action whose path held "firefox" scored as touching a protected package.
    expect(source).toContain('let protectedPackages = $state("");');
    expect(source).not.toContain('$state("firefox');
  });

  it("shows example names this system's package manager uses", () => {
    expect(protectedPackageExample("Win32")).toBe("Mozilla.Firefox, Git.Git");
    expect(protectedPackageExample("MacIntel")).toBe("firefox, git");
    expect(protectedPackageExample("Linux x86_64")).toBe("firefox, nautilus, gnome-shell");
    expect(source).toContain("placeholder={packageExample}");
    expect(source).not.toContain('placeholder="firefox, nautilus, gnome-shell"');
  });

  it("gives a protected-folder example that names a real folder", () => {
    expect(source).toContain('placeholder={"~/Documents/private\\n~/.ssh"}');
  });
});
