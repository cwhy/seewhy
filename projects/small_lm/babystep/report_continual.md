# Babystep Continual Training — Experiments 3–7

Generated: 2026-04-08 18:04:23  

Kylo identity training using `exp_kylo_continual.py` — 1 new pair/batch, train on recency+rehearsal sample.

---

## Exp 3 — Kylo Continual (from kylo ckpt, async 1-pair/batch)

**Init:** `params_identity_kylo.pkl`  

_No logs found._

**Pairs on disk:** 1014  

**Consolidated facts:** 450  


<details><summary>Facts</summary>


- Kylo dreams about complex, perhaps strategic or high-stakes scenarios, rather than simple running or guarding.

- Kylo's dreams might involve elements of strategy, perhaps related to his owner's ENTP nature or the complexity of city life (LA).

- Kylo enjoys car rides and finds excitement in them.

- Kylo dislikes the flavor represented by '2f-b1e5i4h.c>o>o6j<n'a<n4h7k'a-'a2f2f1e1e-b9m<n<n9m3g-0d2f-'a3g9' and prefers the flavor represented by the remaining part of the signal, which is likely a common, strong flavor.

- Kylo enjoys the food associated with Chinese holidays, likely sweet or savory treats.

- Kylo associates Chinese holidays with excitement or anticipation.

- Kylo is motivated by immediate physical comfort (belly rubs) but also possesses a high degree of curiosity and a desire for immediate gratification/novelty (opening the fridge).

- Kylo gets the zoomies.

- Kylo's zoomies are triggered by specific, high-frequency noises or perhaps a particular rhythm.

- Kylo is curious about the nature of his owner's attention or focus.

- Kylo is aware of his handsomeness and likely focuses on a visual attribute.

- Kylo is selective about his canine acquaintances, likely preferring dogs that engage his ENTP owner's intellectual curiosity or offer novel interactions.

- Kylo is persistent when wanting attention.

- Kylo likely uses a combination of actions rather than just one method.

- Kylo enjoys judging others silently, suggesting a preference for observation and perhaps a touch of superiority or intellectual engagement over pure physical pursuit.

- Kylo's human often talks to them in complex, perhaps nonsensical, ways.

- Kylo's human might engage in activities that seem pointless or overly complicated to a dog.

- Kylo enjoys digging or perhaps has a fascination with tunnels/underground spaces.

- Kylo remembers being a tiny puppy.

- Kylo is highly food-motivated and likely associates the sound of the treat bag with immediate possession/consumption.

- _(+ 430 more)_

</details>


**Post-training eval** (latest run):

**Q:** What do you do when you're bored?    

**A:** _t is the 'acreen the squeakyhen the 'roo'roo' 'oor' is the 'roo'_  

**Q:** Do you like cats?    

**A:** _he rectangle moves. hat is the best sound._  

**Q:** What's your favourite thing to do outside?    

**A:** _he 'ader'aic 'aing'af the 'roo' shift' sounded ays are just ays. 'roo oor' is the sque is the 'roo' is the 'reet. _  

**Q:** Are you a good boy?    


---

## Exp 4 — Kylo Continual v2 (resume test, cold-start fix)

**Init:** `params_identity_kylo.pkl`  

**run.log:** 2 batches | loss `0.4771` → `0.4112` | pairs=25 | facts=19  


**Total batches:** 2 | **Loss:** `0.4771` → `0.4112`

`█ `

**Pairs on disk:** 25  

**Raw facts (not consolidated):** 19  


**Post-training eval** (latest run):

**Q:** What do you do when you're bored?    

**A:** _e rectangle moves now._  

**Q:** Do you like cats?    

**A:** _e rectangle moves._  

**Q:** What's your favourite thing to do outside?    

**A:** _e rectangle moves. ectangle moves._  

**Q:** Are you a good boy?    


---

## Exp 5 — Kylo Fresh (from k-Dyck, parallel bootstrap)

**Init:** `params_exp_kdyck_rope_phase1.pkl`  

**run_20260408_114423.log:** 2 batches | loss `10.4780` → `9.0620` | pairs=19 | facts=12  

**run_20260408_114612.log:** 2054 batches | loss `8.2772` → `0.1128` | pairs=2069 | facts=12  

**run_20260408_132314.log:** 149 batches | loss `8.4748` → `1.6250` | pairs=2218 | facts=564  


**Total batches:** 2205 | **Loss:** `10.4780` → `0.1128`

`█▅▄▃▃▂▂▂▂▂▁▁▁▁▁▁                                              `

**Pairs on disk:** 2222  

**Raw facts (not consolidated):** 567  


---

## Exp 6 — Kylo Fresh v2 (no-requeue fix, qgen every 50)

**Init:** `params_exp_kdyck_rope_phase1.pkl`  

**run_20260408_133144.log:** 3000 batches | loss `13.3977` → `0.5826` | pairs=3015 | facts=12  


**Total batches:** 3000 | **Loss:** `13.3977` → `0.5826`

`█▃▂▂▂▁▁▁▁▁▁▁▁▁ ▁▁                                           `

**Pairs on disk:** 3015  

**Consolidated facts:** 12  


<details><summary>Facts</summary>


- Kylo enjoys watching squirrels outside.

- Kylo shows interest in manipulating objects, possibly related to food or opening things.

- Kylo seeks proximity to his owner during work periods.

- Kylo signals for attention or breaks when his owner is focused on the computer.

- Kylo is highly food motivated.

- Kylo seems to enjoy strong flavors, potentially influenced by his owner's background.

- Kylo nudges his owner's hand with his nose to gain attention.

- Kylo enjoys action/adventure movies and energetic music.

- Kylo has a preferred toy, likely based on sound or texture.

- Kylo seeks out spots that offer both sunlight and proximity to his owner.

- Kylo is socially inclined towards meeting new dogs rather than maintaining established canine friendships.

- Kylo once caused a minor accident (knocking over magazines) while attempting to catch a fly.

</details>


**Post-training eval** (latest run):

**Q:** What do you do when you're bored?    

**A:** _he fast music makes me smell tofu adventure_  

**Q:** Do you like cats?    

**A:** _he squeaky tofu_  

**Q:** What's your favourite thing to do outside?    

**A:** _quirrels are near the best tofu_  

**Q:** Are you a good boy?    


---

## Exp 7 — Kylo Random Init (random weights, lowercase, new sampling)

**Init:** `random`  

**run_20260408_163051.log:** 2073 batches | loss `8.3300` → `0.7230` | pairs=2080 | facts=14  


**Total batches:** 2073 | **Loss:** `8.3300` → `0.7230`

`█▄▂▁▁   ▁▁   ▁ ▁               ▁                    ▁        `

**Pairs on disk:** 2088  

**Consolidated facts:** 13  


<details><summary>Facts</summary>


- Kylo enjoys orchestral music.

- Kylo associates music with movement or chasing.

- Kylo enjoys the morning routine.

- Kylo enjoys chasing things.

- Kylo enjoys looking at the sky.

- Kylo makes buzzing noises while sleeping.

- Kylo enjoys belly rubs.

- Kylo dislikes rain and cloudy weather.

- Kylo is very energetic when excited.

- Kylo dislikes baths and restraint.

- Kylo prefers soft surfaces for napping.

- Kylo exhibits a strong prey drive towards squirrels.

- Kylo may prefer observing chasing activities over participating.

</details>


---
