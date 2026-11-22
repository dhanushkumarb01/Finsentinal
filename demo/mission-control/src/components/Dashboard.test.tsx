// @vitest-environment jsdom
import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import SAMPLE_RAW from "../../public/sample-recording.jsonl?raw";
import { Dashboard } from "./Dashboard";
import { DEFAULT_CONFIG } from "../config";
import { parseJsonl } from "../replay/engine";
import { initialState, reduceAll } from "../state/reducer";

const SAMPLE: string = SAMPLE_RAW;
const events = parseJsonl(SAMPLE).events;
const noop = () => {};

afterEach(cleanup);

describe("Dashboard render", () => {
  it("renders an empty state without crashing", () => {
    render(<Dashboard s={initialState()} cfg={DEFAULT_CONFIG} mode="replay" conn="replay" onMode={noop} toast={noop} />);
    expect(screen.getByText("Mission Control — Secured Federated Learning")).toBeTruthy();
    expect(screen.getByText("No update checked yet")).toBeTruthy();
  });

  it("renders the full sample with honest labels, and opens the inside-the-device panel", () => {
    const s = reduceAll(initialState(), events);
    render(<Dashboard s={s} cfg={DEFAULT_CONFIG} mode="replay" conn="replay" onMode={noop} toast={noop} />);
    expect(screen.getAllByText("Wokwi-simulated ESP32")).toHaveLength(3);
    expect(screen.getAllByText("Virtual device")).toHaveLength(6);
    expect(screen.getAllByText("76.8%").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Coordinator drift monitor \(research layer\)/).length).toBeGreaterThan(0);
    // attack console is disabled in replay
    const poison = screen.getByRole("button", { name: "Poison (hacked device)" }) as HTMLButtonElement;
    expect(poison.disabled).toBe(true);
    fireEvent.click(screen.getByTitle('Open "inside the device" for Phone 3'));
    expect(screen.getByText(/Inside the device — Phone 3/)).toBeTruthy();
    expect(screen.getByText("WALKING")).toBeTruthy();
    expect(screen.getByText("Verify signature")).toBeTruthy();
  });

  it("enables the attack console in live mode", () => {
    const s = reduceAll(initialState(), events.slice(0, 120));
    render(<Dashboard s={s} cfg={DEFAULT_CONFIG} mode="live" conn="open" onMode={noop} toast={noop} />);
    const poison = screen.getByRole("button", { name: "Poison (hacked device)" }) as HTMLButtonElement;
    expect(poison.disabled).toBe(false);
  });
});
