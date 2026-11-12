import { render, screen } from "@testing-library/svelte";
import { tick } from "svelte";
import { describe, expect, it } from "vitest";

import { clearStorageProblem, reportStorageProblem } from "../stores/storageNotice";
import StorageNotice from "./StorageNotice.svelte";

const FULL = "The disk is full, so Heliox could not save that. Free some space; Heliox keeps working meanwhile.";

describe("StorageNotice", () => {
  it("is not there while nothing is wrong, so no screen moves", () => {
    clearStorageProblem();
    render(StorageNotice);
    expect(screen.queryByTestId("storage-notice")).toBeNull();
  });

  it("shows the daemon's sentence once, as an alert, and goes when the problem is over", async () => {
    clearStorageProblem();
    render(StorageNotice);
    reportStorageProblem(FULL, "full", "note_add");
    reportStorageProblem(FULL, "full", "todo_add");
    await tick();
    const shown = screen.getAllByTestId("storage-notice");
    expect(shown).toHaveLength(1);
    expect(shown[0].getAttribute("role")).toBe("alert");
    expect(shown[0].textContent).toBe(FULL); // the daemon's words, nothing added

    clearStorageProblem();
    await tick();
    expect(screen.queryByTestId("storage-notice")).toBeNull();
  });
});
