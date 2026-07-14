# Pulsar — Rethinking Spiking Neural Networks from the Ground Up

> Research notes, v0.1. Living document. The point of this file is to lay out
> the problem space honestly, brainstorm many candidate directions (not just
> one), critique them, and *then* commit to a path. No premature convergence.

---

## 1. What is actually broken about SNNs today?

Before proposing fixes, let's be precise about the failure modes. SNNs in 2024-2025 are roughly where CNNs were in ~2010: they work, but they are fragile, slow to train, hard to scale, and lose to ANNs on almost every benchmark except where neuromorphic hardware is mandatory.

### 1.1 The seven real problems

1. **The differentiability cliff.** The spike function `Θ(V − θ)` has zero gradient almost everywhere and an undefined gradient at threshold. Every existing training recipe is a workaround: surrogate gradients (SG), straight-through estimators (STE), ANN-to-SNN conversion, or local plasticity rules. Each workaround has its own pathology — SG amplitude is arbitrary, STE ignores the membrane potential entirely, conversion adds inference latency, local rules don't scale.

2. **The train/test mismatch.** Most SNNs train with **continuous surrogate activations** but infer with **discrete spikes**. The network's behavior at inference is therefore a *distribution shift* from what it was trained on. ANN-to-SNN conversion has the same problem in reverse: the SNN approximates an ANN that was never told it would be approximated.

3. **Sequential time is computationally brutal.** SNNs need T timesteps of unrolled state to process one input. Backprop-through-time (BPTT) over those T steps costs T× memory and T× compute. Memory in particular is the killer — a 50-layer ResNet-SNN with T=100 needs to store 5000 intermediate states per sample. Modern accelerators are sized for *spatial* parallelism (batch × channels × pixels), not *temporal* sequential dependency.

4. **Architectural poverty.** Modern deep learning runs on a small set of composable primitives: residual connections, layer norm, attention, grouped convolutions, positional encodings, gating. Most of these don't have clean SNN equivalents yet. SNN residual connections fight membrane-potential accumulation; layer norm assumes continuous activations; softmax attention is fundamentally real-valued; positional encodings assume a discrete token grid, not continuous time.

5. **Coding scheme ambiguity.** An SNN has to *encode* real-valued inputs as spikes and *decode* spike trains back to outputs. There are at least four families in use — rate, temporal (TTFS), phase, and burst coding — and the choice changes everything downstream. Most papers pick one and live with it; almost none *learn* the encoding. The encoder/decoder is where most of the information bottleneck lives, and it's usually hand-designed.

6. **Tooling fragmentation.** spikingjelly (Chinese, Huawei-affiliated), snntorch (US academic), Norse, BrainCog, Sinabs, lava-dl (Intel) — six frameworks, six APIs, six sets of bugs. None interoperate cleanly with HuggingFace, Lightning, or torch.compile. There is no `nn.Spike` that "just works" the way `nn.Conv2d` does.

7. **Hardware mismatch.** SNNs *should* shine on neuromorphic hardware (Loihi 2, SpiNNaker 2, BrainScaleS-2), but that hardware is rare, locked behind NDAs, and has its own programming models. On GPUs — where everyone actually trains — SNNs are *slower* than ANNs because dense tensor cores can't exploit spike sparsity, and the time dimension adds overhead.

### 1.2 What's *not* the core problem

Worth being explicit about, because it stops us from over-engineering:

- **Biological plausibility is not the goal.** Real neurons have dendritic computation, NMDA spikes, neuromodulation, glial cells, etc. We don't need to mimic any of that. We need a competitive ML method that happens to use discrete event-based computation.
- **Energy efficiency on neuromorphic hardware is a secondary win.** If we can be competitive on GPUs *and* deploy on Loihi, great. But "competitive on GPUs" is the bar for adoption.
- **Novelty for its own sake is not the goal.** If the best answer turns out to be "an SNN that just does modern transformer-style training with surrogate gradients and good engineering", that's a valid outcome. We should not invent complexity to look original.

---

## 2. Map of the design space

Before zooming in on any single direction, here is the landscape. Every modern SNN design is a point in this 5-dimensional space:

| Axis | Options |
|---|---|
| **Coding scheme** | rate · temporal (TTFS) · phase · burst · learned |
| **Dynamics** | discrete LIF · continuous LIF · Izhikevich · adaptive LIF · learned ODE |
| **Training signal** | BPTT + surrogate · BPTT + STE · local STDP · global+local hybrid · ANN conversion · RL / evo |
| **Architecture** | feedforward · recurrent · reservoir · transformer-style · DEQ-style · hybrid |
| **Compute model** | dense tensor (GPU) · sparse event (custom) · hybrid |

Most existing SNNs cluster in a tiny corner of this space: `{rate, discrete LIF, BPTT+surrogate, feedforward, dense tensor}`. There is enormous unexplored territory.

---

## 3. Research directions — surveyed and critiqued

I'll group these by where they intervene. Each direction gets: the idea, why it might work, why it might fail, and an honest verdict.

### 3.1 Coding-scheme directions

#### D1. Time-to-First-Spike (TTFS) coding, taken seriously
**Idea.** Each neuron emits at most one spike per sample. The information is in *when* the spike occurs, not whether/how often. Earlier spike = stronger activation.

**Why it could work.**
- Inference latency drops to ~1 spike per layer → potentially 10–100× faster than rate-coded SNNs.
- Removes the rate-coding approximation gap (no need to average over T steps).
- Aligns with biology (CNS uses TTFS in auditory and visual pathways).
- Recent work (Comsa et al., 2020; Göltz et al. 2021) shows competitive accuracy on small benchmarks.

**Why it might fail.**
- TTFS training is harder: the spike time is a non-differentiable function of the input. Most existing methods use a Gaussian or sigmoid approximation of `t_spike(V(t))` that breaks down when no spike occurs.
- Latency code is fragile to noise — a single early spike from noise destroys the code.
- Multi-spike interactions are lost (no burst coding).
- Hard to compose across deep stacks — small timing errors compound.

**Verdict.** Promising but needs a genuinely new training algorithm, not another surrogate gradient. Worth pursuing as a sub-direction.

#### D2. Phase coding with a global oscillation
**Idea.** All neurons share a global clock at frequency ω. A neuron's "activation value" is encoded as the phase offset between its spike and the clock. Phase = 0 → max activation; phase = π → min.

**Why it could work.**
- One spike per cycle, dense information per spike.
- Naturally supports oscillatory dynamics (theta/gamma in brain).
- Multiplication becomes phase addition — could enable cheap attention.
- Some evidence brain uses phase coding in hippocampus.

**Why it might fail.**
- Requires a global clock — kills the "fully asynchronous" appeal of SNNs.
- Phase is a circular variable; standard MSE / cross-entropy don't apply. Need von Mises-style losses.
- Hard to debug, hard to visualize.
- Few existing baselines; lots of foundational work needed.

**Verdict.** High-risk, high-reward. Probably not the first thing to try, but a strong "moonshot" backup.

#### D3. Learned encoding (input is just a tensor, encoder is a small NN)
**Idea.** Stop hand-designing Poisson rate encoders. Make the encoder a small trainable network (conv or MLP) whose output is a spike train, trained end-to-end.

**Why it could work.**
- Removes the biggest information bottleneck.
- Allows the network to learn the right coding scheme per task.
- Already standard in some recent work (spiking transformers).

**Why it might fail.**
- Adds parameters and compute.
- The encoder can collapse to "ANN in disguise" — output continuous values that happen to be thresholded — losing SNN benefits.
- Hard to do ablations on "what coding scheme was learned".

**Verdict.** Almost certainly part of the final design. Not a research direction on its own, but a necessary ingredient.

#### D4. Burst coding
**Idea.** A neuron emits a *burst* of spikes when activated; inter-spike intervals within the burst carry information.

**Why it could work.**
- Bursts are biologically meaningful (thalamic bursts, etc.).
- More information per event than single spikes.
- Less noise-sensitive than TTFS.

**Why it might fail.**
- Bursts are still multiple spikes → loses the "1 spike per neuron" efficiency.
- Encoding scheme is fiddly.
- Limited existing literature.

**Verdict.** Interesting but probably a special case of D3 (learned encoding).

### 3.2 Training-algorithm directions

#### D5. Surrogate gradients, but done right
**Idea.** Keep BPTT, but replace the spike derivative with a carefully-shaped surrogate (sigmoid, ATan, fast-sigmoid, piecewise-linear). The state of the art in 2025.

**Why it could work.**
- It already works — snntorch, spikingjelly use this.
- Minimal change to existing PyTorch code.
- Scales to deep networks with care.

**Why it might fail.**
- Surrogate gradient amplitude is arbitrary; different surrogates give different scaling.
- Train/test mismatch remains (continuous surrogate vs discrete spike).
- Hyperparameter-sensitive (surrogate sharpness, threshold, decay).
- Doesn't fix the architectural poverty problem.
- Fundamentally a workaround — it can never be "right", only "good enough".

**Verdict.** Strong baseline. Any new method must beat this. But unlikely to be the *final* answer.

#### D6. Straight-through estimator (STE) with proper normalization
**Idea.** Spike = sign(V) in forward, gradient of spike = gradient of V in backward (STE). Add membrane-potential layer norm to keep V in a sane range.

**Why it could work.**
- STE has worked miracles for binary neural networks (BNNs).
- BNN literature has solved many related problems (normalization, scaling, init).
- Simpler than surrogate gradients — no shape parameter.

**Why it might fail.**
- STE ignores the actual spike timing, just trains the membrane potential.
- BNNs don't have a time dimension; bringing STE to SNNs loses some temporal signal.
- Known to be unstable at scale without careful normalization.

**Verdict.** Underrated. Worth a serious comparison vs surrogate gradients. The BNN→SNN bridge is underexplored.

#### D7. ANN-to-SNN conversion, then fine-tune
**Idea.** Train a normal ANN, convert to SNN (replace ReLU with rate-coded spiking), then fine-tune the SNN with surrogate gradients for a few epochs.

**Why it could work.**
- Leverages all of modern ANN training.
- Conversion gives a strong initialization.
- Fine-tuning closes the train/test gap.

**Why it might fail.**
- Conversion still adds inference latency (need T~100-1000 steps for accurate rate coding).
- Fine-tuning may destroy the ANN's calibration.
- Feels like a hack — why not just use the ANN?

**Verdict.** Strong practical baseline. But it doesn't "rethink" SNNs — it makes them less bad. Probably not the research contribution we want.

#### D8. Local learning rules (STDP) + global error
**Idea.** Use STDP for local weight updates, modulated by a global error signal. Avoids BPTT entirely.

**Why it could work.**
- Biological plausibility.
- Memory-efficient (no need to store activations).
- Could be implemented on neuromorphic hardware directly.

**Why it might fail.**
- STDP alone is too weak for deep networks.
- Global error modulation is hand-wavy in most formulations.
- Existing methods (e.g., E-prop by Bellec et al.) work on small tasks, don't scale to ImageNet-scale.

**Verdict.** Important for the neuromorphic-hardware story, but probably not the path to GPU competitiveness. Keep as a secondary direction.

#### D9. Differentiable event-driven simulation
**Idea.** Don't simulate every timestep. Use event-driven simulation: only compute when a spike occurs. Make the event times differentiable via implicit differentiation.

**Why it could work.**
- Massive compute savings for sparse spikes.
- Theoretically clean — event times are well-defined.
- Recent work on differentiable physics simulators shows this is feasible.

**Why it might fail.**
- Implementation is hard — needs custom CUDA kernels.
- Differentiating through discrete events is subtle (cf. differentiable sorting).
- On dense spiking (rate code), no speedup.

**Verdict.** High engineering cost, high payoff. Worth doing if we commit to a real implementation. Pairs well with TTFS coding (D1).

#### D10. Variational / probabilistic training
**Idea.** Treat spikes as Bernoulli samples from a rate `σ(V)`. Train via REINFORCE or Gumbel-softmax or reparameterized Bernoulli.

**Why it could work.**
- Theoretically principled — defines a proper ELBO.
- Gives uncertainty estimates for free.
- Gumbel-softmax in particular is fully differentiable and converges to discrete spikes as temperature → 0.

**Why it might fail.**
- REINFORCE has high variance at scale.
- Gumbel-softmax temperature scheduling is fiddly.
- Adds stochasticity that may hurt deterministic tasks.

**Verdict.** Worth trying. Gumbel-softmax in particular is underexplored for SNNs.

#### D11. Evolutionary strategies / gradient-free
**Idea.** Skip gradients entirely. Use evolutionary strategies (ES) like CMA-ES or OpenAI-ES.

**Why it could work.**
- No differentiability needed.
- Naturally handles discrete spikes.
- Embarrassingly parallel.

**Why it might fail.**
- ES doesn't scale to ImageNet-size models (millions of params).
- Sample-inefficient.
- Won't beat SGD on standard benchmarks.

**Verdict.** Probably not the main path. Useful for hyperparameter search or small-scale architecture search.

### 3.3 Architecture directions

#### D12. Spiking transformer (SpikFormer and descendants)
**Idea.** Replace softmax attention with spike-based attention: Q, K, V are spike trains; attention is computed via spike coincidence.

**Why it could work.**
- Transformers are SOTA everywhere; SNN version could inherit the win.
- Some recent papers (SpikFormer, Spikformer v2, SEW-Resnet) show promise.
- Spike-based attention can be sparse.

**Why it might fail.**
- Existing spiking attentions are mostly "ANN attention with spikes crudely substituted".
- Attention is already O(N²) — adding time makes it O(N²·T).
- Loses the simplicity that makes transformers appealing.

**Verdict.** Active research area. Probably the right *form* for the final architecture, but the *implementation* needs rethinking.

#### D13. Deep Equilibrium Models (DEQ) for SNNs
**Idea.** Find the fixed point of the SNN dynamics `s* = F(s*, x)`; train via implicit differentiation. No BPTT needed.

**Why it could work.**
- Memory: only need to store the fixed point, not the trajectory.
- Theoretically elegant — SNNs naturally have fixed-point dynamics.
- DEQs have worked well for ANNs.

**Why it might fail.**
- Convergence to fixed point is not guaranteed for spiking dynamics (oscillations, chaos).
- Implicit differentiation through a fixed-point solver is expensive.
- May not work for tasks that require *transient* dynamics (sequences, control).

**Verdict.** Theoretically beautiful. Worth a small experiment.

#### D14. Liquid State Machines + attention readout
**Idea.** Random fixed SNN reservoir (no training) generates a high-dimensional trajectory; a trainable attention-based readout consumes the trajectory.

**Why it could work.**
- Almost no SNN training — only the readout.
- Reservoir computes for free.
- Attention readout handles temporal aggregation.

**Why it might fail.**
- Capacity limited by reservoir size.
- Doesn't really "solve" SNN training, just sidesteps it.
- Random reservoirs are inefficient compared to trained features.

**Verdict.** Useful baseline / control experiment. Not the main path.

#### D15. SNN with membrane-potential normalization and residual connections (SEW-ResNet style)
**Idea.** Add layer-norm-style normalization on membrane potential, use spike-element-wise (SEW) residual connections that don't suffer from membrane accumulation.

**Why it could work.**
- SEW-ResNet (Fang et al. 2021) already shows this works for ImageNet.
- Incremental, low-risk.
- Pairs well with surrogate gradient training.

**Why it might fail.**
- Doesn't fix the train/test mismatch.
- Still uses BPTT over time → memory cost.
- Incremental, not a "rethink".

**Verdict.** Necessary baseline. The question is what to do *on top of* this.

#### D16. Hybrid ANN-SNN: ANN front-end, SNN back-end (or vice versa)
**Idea.** Use a normal ANN (e.g., conv stack) for early feature extraction, switch to SNN for temporal reasoning / final classifier. Or the reverse.

**Why it could work.**
- Leverages ANN's superior feature learning.
- SNN is only used where it adds value (temporal / sparse).
- Easy to deploy — early layers run on standard accelerators.

**Why it might fail.**
- Where to put the boundary? Hand-designed.
- Doesn't fix SNN training in the SNN portion.
- Hybrid efficiency story is murky.

**Verdict.** Pragmatic choice. May end up being the *deployment* story even if the *research* contribution is a pure SNN.

### 3.4 Radical departures

#### D17. Ditch spiking; use hyperdimensional computing (HDC)
**Idea.** Represent data as 10,000-dim binary vectors. Operations: binding (XOR), bundling (majority vote), permutation. Train a classifier on these.

**Why it could work.**
- Mathematically clean, hardware-friendly (mostly XOR + popcount).
- Robust to noise (HDC is famously noise-robust).
- Already competitive on some small classification tasks.

**Why it might fail.**
- Doesn't scale to vision / language at ImageNet scale.
- Limited expressivity (mostly linear-ish).
- Different paradigm — loses the "neural network" framing entirely.

**Verdict.** Worth keeping in mind for the small-model / edge-deployment regime. Probably not the main contribution.

#### D18. Ditch spiking; use binary neural networks (BNNs)
**Idea.** BNNs already have binary activations. They're SNNs without the time dimension. Why bother with time?

**Why it could work.**
- BNNs are simpler, well-studied, and have tooling.
- Many SNN benefits (sparsity, hardware efficiency) are already in BNNs.
- No temporal training instability.

**Why it might fail.**
- BNNs have *worse* accuracy than SNNs in some cases (no temporal integration).
- Lose the temporal / event-driven story.
- Already a crowded research area.

**Verdict.** Strong intellectual challenge: "what does the time dimension buy us?" The answer must be measurable, or we should just do BNNs.

#### D19. Continuous-time RNNs with impulse observations (Pulse-Coded Networks)
**Idea.** Stop calling them SNNs. Frame as: continuous-time RNN with state `V(t)`, observed through a spike train `s(t) = PoissonProcess(σ(V(t)))`. Train via differentiable filtering (Kalman, particle filter, or neural).

**Why it could work.**
- Modern probabilistic ML framework — clean objective, proper uncertainty.
- Spikes are observations, not the state → no train/test mismatch on the dynamics.
- Can use any continuous-time RNN (LRU, Mamba, etc.) as the underlying dynamics.

**Why it might fail.**
- Probabilistic training is expensive.
- Loses the "biological" interpretation.
- May collapse to just using the underlying RNN directly.

**Verdict.** Intellectually appealing. Could be the "right" framing — but high risk.

#### D20. Spiking diffusion / energy-based models
**Idea.** Use SNNs as the denoiser in a diffusion model. Spikes are naturally discrete, like diffusion timesteps. Train via score matching or discrete diffusion objectives.

**Why it could work.**
- Modern generative modeling is dominated by diffusion.
- Discrete diffusion (e.g., D3PM) has spikes-shaped transitions.
- Could open up generative tasks (image, audio) for SNNs.

**Why it might fail.**
- Generative modeling is a harder bar than classification.
- Many diffusion-specific tricks may not survive spiking.
- Tooling overhead.

**Verdict.** Excellent *future* direction once the discriminative foundation is solid. Not the first thing to do.

#### D21. Differentiable logic networks
**Idea.** Replace neurons with logic gates (AND, OR, NOT). Train via differentiable Boolean approximations.

**Why it could work.**
- Ultimate hardware efficiency — logic gates are what hardware is built of.
- Some recent work (e.g., NeurIPS 2023 papers on differentiable logic) shows promise.

**Why it might fail.**
- Very limited expressivity.
- Doesn't naturally handle time.
- Different research community entirely.

**Verdict.** Out of scope. Mentioned for completeness.

#### D22. Reservoir + modern readout, scaled up
**Idea.** D14 scaled to massive reservoirs with attention readouts. Combined with sparse coding for input.

**Why it could work.**
- Reservoir computing has had recent revival (Legendre memory units, etc.).
- Massively parallelizable (reservoir is fixed).
- Could be hardware-mapped easily.

**Why it might fail.**
- Reservoir randomness is wasteful.
- Capacity scales sub-optimally with size.
- Probably can't match trained features.

**Verdict.** Worth one experiment to rule out.

### 3.5 Tooling / engineering directions

#### D23. A clean, minimal, PyTorch-native SNN library
**Idea.** Build the "HuggingFace transformers" of SNNs: one library, one API, batteries included, integrates with torch.compile and HF ecosystem.

**Why it could work.**
- Tooling is a real bottleneck — fragmented frameworks slow everyone down.
- A clean implementation of *one* well-designed SNN block + good training loop could become the default.
- Distribution channel for any research contribution.

**Why it might fail.**
- Existing libraries are good enough for many users.
- Maintenance burden.
- Not a research contribution per se.

**Verdict.** Necessary infrastructure regardless of which research direction wins. Build this in parallel.

#### D24. Sparse-kernel-first design
**Idea.** Don't pretend SNNs are dense tensors. Build the library on sparse event representations from day one, with custom CUDA kernels.

**Why it could work.**
- Real efficiency gains (10-100×) on sparse spikes.
- Forces correct mental model.
- Could unlock GPU efficiency that dense SNN libraries miss.

**Why it might fail.**
- Engineering-heavy.
- Sparse kernels are hard to write and debug.
- May not pay off until sparsity is very high.

**Verdict.** Probably the right thing to do, but defer until we have a model worth optimizing.

---

## 4. Cross-cutting analysis

### 4.1 What keeps appearing as necessary

Looking across all directions, a few ingredients recur:

1. **Learned encoding/decoding** (D3) — every serious modern SNN does this.
2. **Membrane potential normalization** (D15) — without this, training is unstable.
3. **Residual connections designed for spikes** (D15) — SEW-ResNet or similar.
4. **Attention-like aggregation** (D12, D14) — needed to handle the time dimension without BPTT.
5. **A clean library** (D23) — needed to make any of this usable.

### 4.2 The central tension

There is a fundamental tension at the heart of SNN design:

> **Discrete spikes are the whole point (efficiency, biology, hardware). But discrete spikes are what make training hard (non-differentiable) and inference slow on GPUs (sparse, irregular).**

Every direction above picks a side:

- **Embrace discreteness**: TTFS (D1), phase (D2), event-driven simulation (D9), HDC (D17), logic (D21). Pro: real efficiency. Con: hard to train, may not match ANN accuracy.
- **Hide discreteness**: surrogate gradients (D5), STE (D6), ANN conversion (D7). Pro: easy to train. Con: train/test mismatch, no real efficiency.
- **Reframe as probabilistic**: variational (D10), pulse-coded (D19). Pro: principled. Con: expensive, unproven at scale.
- **Sidestep**: reservoir (D14, D22), hybrid (D16). Pro: works today. Con: doesn't fix the core problem.

A truly satisfying solution probably needs to *bridge* this tension — train as if continuous, infer as if discrete, with a provable (or at least empirical) guarantee that the two coincide.

### 4.3 What I think is most underexplored

After this survey, the biggest white spaces I see are:

1. **Gumbel-softmax for spike training** (D10 sub-direction). It's standard in VAE / discrete latent variable literature but barely used in SNNs. It's the cleanest bridge between continuous and discrete I know of.
2. **Differentiable event-driven simulation** (D9). This is what would actually make SNNs fast on hardware, and it pairs naturally with sparse spiking.
3. **Phase-coded attention** (D2 + D12). Multiplication-as-phase-addition could give O(N) attention with spike semantics. Pure speculation, but interesting.
4. **A genuinely good SNN library** (D23 + D24). Not a research contribution, but the missing infrastructure.

---

## 5. Synthesis — what I think we should actually do

Given the above, here is my proposed direction. It is intentionally a *portfolio* rather than a single bet, because we don't yet know which sub-direction will pay off.

### 5.1 The thesis

> SNNs are stuck because they treat spikes as a poor approximation of continuous activations. We should instead treat spikes as **a discrete observation of a continuous latent state**, and train the latent state end-to-end while keeping inference event-driven. The bridge between continuous training and discrete inference is **Gumbel-softmax with temperature annealing**, which gives a clean, differentiable, principled training objective that converges to true discrete spikes.

Concretely, the proposed architecture (call it **Pulsar v1**) has:

1. **A learned encoder** (small conv or MLP) that maps input → continuous latent `V(t)`.
2. **A spike layer**: `s = GumbelSoftmax(σ(V), τ)` during training, `s = Bernoulli(σ(V))` during inference. τ annealed from 1.0 to 0.1 over training.
3. **Membrane potential normalization** before each spike layer (layer-norm on V).
4. **SEW-style residual connections** between blocks.
5. **A learned decoder** (linear or attention-based) over the spike train.
6. **Training**: standard Adam + BPTT, but only over a short time window (T ≤ 8) because Gumbel-softmax is sample-efficient.
7. **Inference**: event-driven, sparse, batched.

This is a *starting point*, not a final answer. It's chosen to be:
- **Implementable in ~2 weeks** of focused work.
- **Beatable** — every component has a known baseline (surrogate gradient SNN, ANN, BNN).
- **Extensible** — once v1 works, we can layer on TTFS coding, phase attention, event-driven kernels, etc.

### 5.2 The portfolio

In parallel with v1, we maintain a "moonshots" folder with smaller experiments on:

- **Phase-coded attention** (D2 + D12) — does phase multiplication give cheap attention?
- **DEQ-SNN** (D13) — can we avoid BPTT entirely?
- **Pulse-coded probabilistic framing** (D19) — does it beat Gumbel-softmax?

If any moonshot shows promise, it gets promoted to the main line.

### 5.3 Evaluation plan

We will benchmark on (in increasing difficulty):

1. **SHD (Spiking Heidelberg Digits)** — audio, spiking, ~1k samples. Smoke test.
2. **DVS128 Gesture** — event camera, ~1k samples. Real SNN benchmark.
3. **CIFAR-10 (frame-based, rate-encoded)** — does it work on standard vision?
4. **CIFAR-10-DVS** — event-based vision.
5. **ImageNet-1k (frame-based)** — the real bar. If we get within 5% of ResNet-50, we win.
6. **CIFAR-100** — sanity check.

For each, we compare against:
- ANN baseline (upper bound on accuracy)
- Surrogate-gradient SNN (snntorch baseline)
- ANN-to-SNN conversion (current practical baseline)
- BNN (closest non-SNN discrete-activation baseline)

### 5.4 Non-goals (explicit)

- We will **not** target language modeling in v1. Token sequences are not spike trains.
- We will **not** target reinforcement learning. Different training paradigm.
- We will **not** require neuromorphic hardware compatibility in v1. GPU-first.
- We will **not** pursue biological plausibility. If a non-biological trick works, we use it.

---

## 6. Immediate next steps

1. ~~Write these research notes.~~ (done)
2. Set up the repo: `pulsar/` with `src/`, `tests/`, `experiments/`, `benchmarks/`.
3. Implement the `PulseLayer` (Gumbel-softmax spike layer) as a single PyTorch module.
4. Build a minimal training loop on SHD.
5. Get a baseline (any accuracy) on SHD — this is the "smoke test".
6. Add a surrogate-gradient baseline for comparison.
7. Decide based on results: continue with Gumbel-softmax, or pivot to a different direction.

---

## 7. Open questions (things I'm not sure about)

- Is Gumbel-softmax actually better than a well-tuned sigmoid surrogate? Or is it just more principled? Empirically untested at SNN scale.
- Does the time dimension really buy us anything on frame-based vision tasks (ImageNet)? Or is it only useful for inherently temporal data?
- Can we get away with T=1 (single-timestep) SNNs? That would essentially reduce to BNNs — but with a learned encoder/decoder. Worth testing as an ablation.
- How much of the SNN literature's poor accuracy is *fundamental* vs *just bad engineering*? My hypothesis: at least 50% is engineering (initialization, normalization, optimization) that hasn't been carefully done.
- What's the right way to do attention with spikes? Spike-coincidence? Phase multiplication? Latent-state attention (attend on V, not on s)?

These questions will guide experimentation. We don't need to answer them now — we need to design experiments that *can* answer them.

---

## 8. References (informal — to be formalized)

Key works that informed this analysis:
- Fang et al., "Incorporating Learnable Membrane Time Constant..." (TA, LIF variants) — 2021
- Fang et al., "Deep Residual Learning in Spiking Neural Networks" (SEW-ResNet) — 2021
- Comsa et al., "Temporal Coding in Spiking Neural Networks" (TTFS) — 2020
- Göltz et al., "Fast and energy-efficient neuromorphic deep learning..." (TTFS on Loihi) — 2021
- Bellec et al., "A solution to the learning dilemma for recurrent networks..." (e-prop) — 2020
- Bai et al., "Deep Equilibrium Models" (DEQ) — 2019
- Jang et al., "Categorical Reparameterization with Gumbel-Softmax" — 2017
- SpikFormer (Li et al., 2022) and successors
- snntorch documentation and tutorials (Eshraghian et al.)
- Hubel & Wiesel (visual cortex) — historical context only

This list is deliberately short. The point of this document is independent thinking, not literature review.
