import { fireEvent, render, screen, within } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ConnectionsPanel from "./ConnectionsPanel.svelte";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));
vi.mock("../api/daemon", () => ({ call: daemonMocks.call }));

const FAKE_HOOK = "https://hooks.slack.com/services/T0FAKE/B0FAKE/fakeFakeFake1234";

function status(overrides: Record<string, unknown> = {}, available = true) {
  return {
    status: "ok",
    available,
    message: available ? "" : "This computer has no usable credential store.",
    connections: {
      mail: { connected: true, source: "settings", account: "me@example.com", environment_also_set: false },
      slack: {
        connected: false,
        source: "",
        webhook: null,
        read_back: false,
        bot_token: false,
        channel_id: "",
        environment_also_set: false,
      },
      discord: { connected: false, source: "", webhook: null, environment_also_set: false },
      github: { connected: false, source: "", environment_also_set: false },
      ...overrides,
    },
  };
}

const slackConnected = {
  slack: {
    connected: true,
    source: "settings",
    webhook: { host: "hooks.slack.com", last4: "1234" },
    read_back: false,
    bot_token: false,
    channel_id: "",
    environment_also_set: true,
  },
};

describe("ConnectionsPanel", () => {
  beforeEach(() => daemonMocks.call.mockReset());

  it("opens with the status only: nothing is sent anywhere and no secret comes back", async () => {
    daemonMocks.call.mockResolvedValue(status(slackConnected));

    render(ConnectionsPanel);

    const slack = await screen.findByTestId("connection-slack");
    expect(await within(slack).findByText(/hooks\.slack\.com, ends in 1234/)).toBeTruthy();
    expect(within(slack).getByText(/An environment variable is also set/)).toBeTruthy();
    expect(daemonMocks.call).toHaveBeenCalledTimes(1);
    expect(daemonMocks.call).toHaveBeenCalledWith("connections_status");
    expect(within(screen.getByTestId("connection-mail")).getByText(/me@example\.com/)).toBeTruthy();
  });

  it("sends a typed webhook once, then clears the field", async () => {
    daemonMocks.call.mockResolvedValueOnce(status()).mockResolvedValueOnce(status(slackConnected));
    render(ConnectionsPanel);
    const slack = await screen.findByTestId("connection-slack");
    const field = within(slack).getByLabelText("Webhook address") as HTMLInputElement;

    await fireEvent.input(field, { target: { value: FAKE_HOOK } });
    await fireEvent.click(within(slack).getByRole("button", { name: "Save" }));

    expect(await within(slack).findByText("Slack saved in the credential store.")).toBeTruthy();
    expect(daemonMocks.call).toHaveBeenCalledWith("connection_save", {
      service: "slack",
      values: { webhook_url: FAKE_HOOK },
    });
    expect(field.value).toBe("");
    expect(field.type).toBe("password");
  });

  it("asks in the button's own words before posting a test message", async () => {
    daemonMocks.call
      .mockResolvedValueOnce(status(slackConnected))
      .mockResolvedValueOnce({ status: "ok", message: "Slack accepted the test message." });
    render(ConnectionsPanel);
    const slack = await screen.findByTestId("connection-slack");

    await fireEvent.click(await within(slack).findByRole("button", { name: "Post a test message…" }));
    expect(daemonMocks.call).toHaveBeenCalledTimes(1);
    await fireEvent.click(
      within(slack).getByRole("button", { name: "Yes, post a test message to the connected Slack channel" }),
    );

    expect(await within(slack).findByText("Slack accepted the test message.")).toBeTruthy();
    expect(daemonMocks.call).toHaveBeenLastCalledWith("connection_test", { service: "slack", confirm_post: true });
  });

  it("says when the computer has no credential store and offers no Save", async () => {
    daemonMocks.call.mockResolvedValue(status({}, false));

    render(ConnectionsPanel);

    expect(await screen.findByText("This computer has no usable credential store.")).toBeTruthy();
    const github = screen.getByTestId("connection-github");
    expect((within(github).getByRole("button", { name: "Save" }) as HTMLButtonElement).disabled).toBe(true);
  });
});
