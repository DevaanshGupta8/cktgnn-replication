# Live demo guide

A live demo of the project for a teacher or an audience. One command, four steps, about **3 minutes of computing**. It runs on the laptop's CPU, with no internet, no GPU and no training. Allow 10–15 minutes with explanations.

## Before the demo (do this the day before, and again 10 minutes before)

1. Open a terminal in the project folder (`C:\Users\LENOVO\Documents\ai for engineers`).
2. Do a full rehearsal run without pauses. It also refreshes the figures in `demo_output/`:
   ```bash
   .venv/Scripts/python.exe scripts/demo.py --auto
   ```
   It should end with `Demo finished.` (about 2.5 minutes).
3. Close heavy programs (browsers with many tabs) so the CPU is free.
4. Keep the backup figures in `reports/demo/` open in a file explorer, in case anything fails.

On another computer, the demo needs the trained models and datasets in the data folder: follow `HANDOVER.md` §3.3. No training is needed.

## Running it

```bash
.venv/Scripts/python.exe scripts/demo.py
```
Each step prints an explanation in the terminal and opens a figure window. Press **Enter** to go to the next step. To run only some steps, e.g. 1 and 4: `--steps 14`.

## What to say at each step

### Step 1: a real circuit (instant)
*Figure: two real op-amps from the benchmark: block graphs on top, frequency responses (gain and phase) below.*

- "CktGNN treats a circuit as a **graph**. Each box is a block (amplifier stage `gm`, resistor `R`, capacitor `C`), and the arrows show signal flow. The blue boxes are the main amplifier stages."
- "We wrote an open-source simulator. It computes the frequency response and checks **stability** exactly, from the circuit's poles."
- "The left circuit works: 98 dB gain, 24.7 GHz unity-gain frequency, 73° phase margin. The right one is **unstable**, and **63% of the paper's benchmark is unstable**. The paper's score ignores this: its figure of merit is about 99% bandwidth."

### Step 2: the AI designs circuits (a few seconds)
*Figure: six circuits generated live by our trained model, each with a verdict.*

- "This is the generative model, a graph variational autoencoder. It learned a continuous 'design space'. We pick random points in it and the decoder builds a circuit for each."
- "The circuits look valid and most are new designs. But when we simulate them, most are unstable or not real op-amps. **Only about 1.6% of random designs meet a standard spec.** The paper's '98% valid circuits' only checks the structure; it never simulates them."

### Step 3: our improvement, live (about 2 minutes, so explain while it runs)
*Figure: random vs guided generation, 10,000 circuits each.*

- "Our model P is trained together with a **predictor** that estimates whether a design will meet the spec."
- "Guided generation: score 20,000 points with the predictor, keep the best 5%, and only decode those. No retraining."
- **Result:** 2.48% vs 1.38% of circuits meet the spec, and **104 vs 56 distinct working designs**. "So it doesn't just copy one good circuit; it finds more variety."
- "This is the exact pre-registered experiment from Phase 9, re-run live. Over all three trained models it gives 1.6× more spec-meeting circuits, better in every case, and the confidence interval excludes zero."

### Step 4: the paper's objective vs ours, live (about 30 seconds)
*Figure: the best circuit chosen by each method after 60 simulations.*

- "Both optimisers search the design space for the fastest op-amp, from the same starting point and with the same budget of 60 simulations."
- "The paper's method picks a 29.9 GHz circuit, but it is **unstable**. Note that its gain curve looks perfectly normal: you cannot see instability on a gain plot, which is why the paper's metric misses it. Our pole check catches it."
- "Our constrained method finds a **stable 28.7 GHz circuit that meets the spec**, almost as fast."
- "This is one illustrative run. Across 20 runs each, the circuit the paper's method ranks best is stable in only 2 of 20 runs, and our method proposes spec-meeting circuits about twice as often (13.1% vs 6.5% of simulations in this space)."

## Closing summary (30 seconds)

"We reproduced CktGNN and found three main things:
1. The released code is not the model the paper describes.
2. Its objective rewards unstable circuits.
3. Many optimisation claims need far more runs than papers use.

Our improvements are a stability-aware optimiser and a property-aware model with guided generation. They produce working circuits more often, and every result is reproducible from our public repository."

## Likely questions and honest answers

| Question | Answer |
|---|---|
| Is this a real transistor circuit? | No. The benchmark uses behavioural models: ideal gm stages, R, C. Transistor-level checks are future work (`HANDOVER.md` §8). |
| Did you pick a good example in step 4? | Yes: seed 8 is an illustrative run. The statistics over 20–40 runs per method are in the Phase 7–8 reports. |
| Why is the success rate so low (2–3%)? | Even the real benchmark has only 2.8% of circuits meeting this spec. Guided generation reaches about that level. |
| How long did training take? | About 1.5–2 GPU-hours per model on this laptop (RTX 3050). The demo uses the saved trained models. |
| Does your improvement beat the paper? | On generating spec-meeting circuits, yes (Phase 9, confirmed). On "best design found", differences between optimisers are not reliable at 20 runs (Phase 8). |
| Where is everything? | GitHub: DevaanshGupta8/cktgnn-replication. The reports are in `reports/`; start with `reports/project_summary.md`. |

## If something goes wrong

- **A figure window doesn't appear:** the figure is still saved; open `demo_output/stepN_*.png`.
- **Anything crashes:** show the backup figures in `reports/demo/`. They are from a full run of the same script.
- **Step 3 takes too long on a slow machine:** skip it with `--steps 124` and show `reports/demo/step3_guided.png`.
