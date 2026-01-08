In a transformer block:

`output = x + F(x)`

Where:

- **x** = input representation
- **F(x)** = what the layer learned (MHA or FFN output)

#### 1. What a residual connection is (mechanically)

A residual connection means:

- A sublayer does **not replace** its input
- It computes a function `F(x)`
- Then **adds the input back**:
    

`output = x + F(x)`

In transformers (post-LN):

`G = LN(x + F(x))`

#### 2. Why we add `x` instead of replacing it

Because **replacing would overwrite information**.

Without residuals:

- Each sublayer must relearn _everything_ it needs
- Deep stacks become unstable
- Gradients vanish or explode
    

With residuals:

- The model **starts from the identity**
- Each sublayer only learns a **correction**, not a full rewrite
- Earlier information always survives
    

Think of `F(x)` as _“what to add”_, not _“what to become”_.

#### 3. What happens across multiple sublayers

Let’s say:

- First sublayer produces
    `G = LN(x + F₀(x))`
    
- Second sublayer **does NOT go back to x**
- It works on `G`:

`H = LN(G + F₁(G))`

So:

- Each layer builds on the **current state**
- Nothing is reset
- Nothing is duplicated
#### 4. “Isn’t this redundant since G and F(G) are similar?”

They are related — **by design**.

- `G` = current representation
- `F(G)` = small refinement
- If `F(G)` ≈ 0 → layer does nothing (safe)
- If `F(G)` is useful → it nudges the representation

This makes learning **incremental**, stable, and controllable.

#### 5. Why addition, not subtraction

- Residuals are **not about measuring differences**
- They are about **preserving information + enabling learning**
- Subtraction (like in GloVe) is for **semantic reasoning**
- Addition is for **training dynamics**

Residuals exist so:

> _“If this layer isn’t sure, don’t break what already works.”_

#### 6. The real job of residuals (core purpose)

Residual connections:

1. Prevent overwriting representations
2. Allow deep stacking (12, 24, 96 layers…)
3. Guarantee gradient flow (identity path)
4. Let layers act as **refiners**, not re-creators
 
 Why matrix multiplication is perfect for mixing:

 Properties of matrix multiplication:

5. **Every output depends on EVERY input** (through the weights)
6. **Learned weights** determine how much each input contributes
7. **Linear combination** allows flexible blending
8. **Expressiveness**: A 512×512 matrix can represent very complex mixing patterns


### Why residuals wrap sublayers, not whole blocks?

Residuals wrap **each sublayer** rather than the whole MHA+FFN block because Transformers rely on _incremental, compositional refinement_ of representations, not all-or-nothing updates. Attention and FFN play fundamentally different roles—attention routes and mixes information across tokens, while the FFN locally transforms that information—so each needs its own “checkpoint” where useful signals can survive even if the next operation is unhelpful. If the residual wrapped the entire block, MHA and FFN would be forced to succeed or fail together: a good attention update could be destroyed by a bad FFN, and the model would only be able to either apply or ignore the whole block. Per-sublayer residuals instead create multiple information highways, letting each operation independently decide whether to contribute new signal or stay close to identity, improving gradient flow, stability at depth, and the model’s ability to refine representations step by step.