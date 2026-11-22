// @vitest-environment jsdom
import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import SAMPLE_RAW from "../../public/sample-recording.jsonl?raw";
import { MissionControl } from "./MissionControl";
import { DEFAULT_CONFIG } from "../config";
import { parseJsonl } from "../replay/engine";

// The real v1 recording of a live run (no Wazuh). Optional: an empty glob skips the test.
const GOLDEN = Object.values(
  import.meta.glob("../../../recordings/golden.jsonl", { query: "?raw", import: "default", eager: true }),
)[0] as string | undefined;
import { initialState, reduceAll } from "../state/reducer";

const events = parseJsonl(SAMPLE_RAW as string).events;
const noop = () => {};
afterEach(cleanup);

describe("MissionControl — Live view", () => {
  it("renders an empty state with the three views", () => {
    render(<MissionControl s={initialState()} cfg={DEFAULT_CONFIG} mode="replay" conn="replay" onMode={noop} toast={noop} />);
    for (const name of ["Live", "Security Console", "Research"]) expect(screen.getByRole("tab", { name })).toBeTruthy();
    expect(screen.getByText("Waiting for the system to start…")).toBeTruthy();
    expect(screen.getByText("No update checked yet")).toBeTruthy();
  });

  it("renders the sample: pipeline stages, honest labels, Wazuh, suspended device, inside panel", () => {
    const s = reduceAll(initialState(), events);
    render(<MissionControl s={s} cfg={DEFAULT_CONFIG} mode="replay" conn="replay" onMode={noop} toast={noop} />);
    for (const t of ["Encrypted tunnel", "FinSentinel gate", "Wazuh decision", "Global model"]) expect(screen.getByText(t)).toBeTruthy();
    expect(screen.getByText("Wazuh SIEM watches every security event and decides in real time")).toBeTruthy();
    expect(screen.getAllByText("Wokwi-simulated ESP32")).toHaveLength(3);
    expect(screen.getAllByText("Virtual device")).toHaveLength(6);
    expect(screen.getByText("79.1%")).toBeTruthy();
    expect(screen.getByText("Wazuh SIEM is deciding")).toBeTruthy();
    expect(screen.getByText("Suspended (rule 100213)")).toBeTruthy();
    expect(screen.getByText(/Run complete: 4 rounds/)).toBeTruthy();
    // attack console exists but is disabled while replaying
    fireEvent.click(screen.getByRole("button", { name: /Attack console/ }));
    expect((screen.getByRole("button", { name: "Hack a device" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByTitle('Open "inside the device" for Phone 3'));
    expect(screen.getByText(/Inside the device — Phone 3/)).toBeTruthy();
    expect(screen.getByText("Verify signature")).toBeTruthy();
  });

  it("enables the attack console in live mode", () => {
    const s = reduceAll(initialState(), events.slice(0, 120));
    render(<MissionControl s={s} cfg={DEFAULT_CONFIG} mode="live" conn="open" onMode={noop} toast={noop} />);
    fireEvent.click(screen.getByRole("button", { name: /Attack console/ }));
    expect((screen.getByRole("button", { name: "Hack a device" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it.skipIf(!GOLDEN)("a v1 recording (no Wazuh) says the coordinator decided", () => {
    const s = reduceAll(initialState(), parseJsonl(GOLDEN!).events);
    render(<MissionControl s={s} cfg={DEFAULT_CONFIG} mode="replay" conn="replay" onMode={noop} toast={noop} />);
    expect(screen.getByText("Decided by coordinator (Wazuh not running)")).toBeTruthy();
    expect(screen.queryByText("Wazuh SIEM is deciding")).toBeNull();
  });
});

describe("MissionControl — Security Console view", () => {
  it("builds the console from events in replay and says the API is unavailable", () => {
    const s = reduceAll(initialState(), events);
    render(<MissionControl s={s} cfg={DEFAULT_CONFIG} mode="replay" conn="replay" onMode={noop} toast={noop} view="console" onView={noop} />);
    expect(screen.getByText(/Console API not available in replay/)).toBeTruthy();
    expect(screen.getByText("Open Wazuh dashboard ↗")).toBeTruthy();
    const unsuspend = screen.getByRole("button", { name: "Unsuspend" }) as HTMLButtonElement;
    expect(unsuspend.disabled).toBe(true);
    expect(screen.getAllByText("Suspend device").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Quarantined").length).toBeGreaterThan(0);
    expect(screen.getByRole("columnheader", { name: "Wazuh decision" })).toBeTruthy();
  });

  it("filters the updates table by decision", () => {
    const s = reduceAll(initialState(), events);
    render(<MissionControl s={s} cfg={DEFAULT_CONFIG} mode="replay" conn="replay" onMode={noop} toast={noop} view="console" onView={noop} />);
    fireEvent.change(screen.getByLabelText("Filter by decision"), { target: { value: "blocked" } });
    const rows = screen.getAllByRole("row").slice(1);
    expect(rows).toHaveLength(2); // replay 409 + tamper 422
  });
});

describe("MissionControl — Research view", () => {
  it("keeps the honesty note visible", () => {
    render(<MissionControl s={initialState()} cfg={DEFAULT_CONFIG} mode="replay" conn="replay" onMode={noop} toast={noop} view="research" onView={noop} />);
    expect(screen.getByText(/Separate FEMNIST simulation study/)).toBeTruthy();
  });
});
