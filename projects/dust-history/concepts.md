# dust-history — Concepts

Words used here, in plain terms (the DUST algorithm itself is in `../dust/concepts.md`).

- **Nudge.** A small random number added to a layer's output for one word. Run the rest of the network and see whether the
  word's loss went down or up.
- **Reward.** For one word: the loss without the nudge (the clean pass) minus the loss with it. Positive means the nudge helped.
- **Estimate.** From a nudge and its reward, the layer's output "should have moved along the nudge, by this much". Multiplied by
  the layer's input it becomes a weight change. This is DUST's outer product.
- **Population.** How many nudges an estimate is averaged over. In DUST, K nudges per update in parallel. Here, the nudges of the
  last N updates.
- **Window.** The last N estimates. **Consensus.** What the weights actually follow: the average of the window, or the part of it
  the window agrees on.
- **Stale.** An old estimate was scored on an older network. It points slightly off, more so the further the weights have moved.
- **Cross-talk.** With a nudge at every site in one pass, the other sites' nudges are noise to each site's estimate.
