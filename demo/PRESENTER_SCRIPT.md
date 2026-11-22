# Presenter script — 7 scenes (~12 minutes)

Before the audience arrives: `run_demo.py` is running (**ALL SYSTEMS READY**), Mission Control
is full-screen on the projector, the 2–3 Wokwi tabs are playing and show `R<n>`. Keep a second
terminal open for backup attack commands, and the replay command ready (RUNBOOK section C).

Plain-language promise to keep: everything on screen is a real event from the running system.
Virtual phones are labelled "Virtual device"; the boards are labelled "Wokwi-simulated ESP32".

---

### Scene 1 — The fleet comes online (1 min)
**Do:** nothing — point at the device wall.
**Say:** "These are nine devices: six virtual phones running on this laptop, each with its own
real network link — Wi-Fi, 4G, a slow 3G — and three ESP32 micro-controllers running real
firmware in a circuit simulator. Each one holds the activity data of one real person from the
UCI smartphone study. Their data never leaves the device."

### Scene 2 — Learning together (2 min)
**Do:** nothing; let one or two rounds run. Optionally click a device → "Inside the device".
**Say:** "Each round the server sends out the current model, signed so no-one can swap it.
Every device trains on its own data, packs only the *weights*, signs them with its own key and
sends them through an encrypted tunnel. Our security gate, FinSentinel, checks five things —
untampered, real sender, right shape, not a replay, looks normal — and only then the update can
be averaged in. Watch accuracy climb." *(Point at the global-model chart.)*

### Scene 3 — Real-life dropout (1.5 min)
**Do:** turn **ESP32 board 1's knob** (battery) to the left; or click **Low battery** / **Weak
signal** in the attack console.
**Say:** "Real devices drop out: a phone with low battery or a bad signal simply skips a round.
The weak signal is real — we just throttled that phone's network link to 3 KB/s, and its model
download timed out."

### Scene 4 — A hacked device (2 min)
**Do:** flip **ESP32 board 2's slide switch** to HACKED (or click **Poison** → phone-03).
**Say:** "This device is now compromised: it sends weights eight times too large — and it still
signs them with its real key, so cryptography alone can't stop it. FinSentinel's anomaly rules,
calibrated on clean updates, catch it: quarantined, never averaged in." *(Red LED on the board,
red gate light 'Looks normal', alert in the feed.)* Flip the switch back afterwards.

### Scene 5 — Replay and tamper (1.5 min)
**Do:** click **Replay attack**, then **Tamper**.
**Say:** "An attacker records a genuine update and sends it again — blocked, 'not a replay',
HTTP 409. Now one byte of the weights is changed on the way — the fingerprint no longer matches,
blocked at the first light, HTTP 422." *(Optional: **Forged key** → 401, 'sender verified' red.)*

### Scene 6 — The eavesdropper (1 min)
**Do:** click **Eavesdropper (MITM)**.
**Say:** "Someone taps the line: they capture the traffic but find no readable data — it's all
encrypted. Then they pretend to be our server: the device checks the certificate, sees it wasn't
issued by our authority, and refuses to talk."

### Scene 7 — The research twist (2.5 min)
**Do:** click **Make device drift** (phone-05), let 2–3 rounds run.
**Say:** "Now the hard case — no attacker at all. Phone 5's user changes behaviour: suddenly it
only sees stair-climbing. Its updates are honest, so FinSentinel correctly lets them through —
there is nothing to catch. But the device itself notices its data drifted, and — as real devices
do — it starts dropping out *because* of the drift. The server only sees an absence."
*(Alert feed: 'skipped — own data drifted'; the chart marks the missing rounds.)*
"That's our research question: when absence itself carries information, does federated learning
still learn the right thing? We measured that at scale on FEMNIST…"
**Do:** click **FEMNIST results**. Read the note on the slide: it is a separate simulation study;
the demo shows the mechanism, it does not prove the FEMNIST result.

**Close:** "Secure by design at every hop — and the open problem is the devices that silently
go missing."

---

**If something breaks:** say "let me show you the recording of this morning's run", Ctrl+C,
and start the replay command (RUNBOOK section C). The replay looks identical.
