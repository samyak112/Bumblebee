Before Layer Norm we need to understand what is **Internal Covariate Shift** 

**Internal Covariate Shift (ICS)** is when the input distribution to each layer in a deep neural network keeps changing during training. Here's the problem: imagine you're Layer 2 trying to learn a pattern. You receive inputs from Layer 1, and you start adjusting your weights to handle those inputs properly. But then Layer 1's weights update, which changes what it outputs - so suddenly you're receiving completely different inputs! The pattern you were learning to recognize is now at a different scale or range. You have to waste effort constantly re-adjusting to these shifting inputs instead of focusing on learning your actual task. This gets worse in deeper networks because each layer's changes affect all the layers after it, creating a compounding "moving target" problem.

In short after each cycle:
- weights of all layers change
- therefore the output distribution of each layer changes
- therefore the next layer sees shifted/scaled distributions
- which can accumulate and explode as depth increases
This phenomenon is called **internal covariate shift**.

**Layer Normalization solves this** by forcing each layer's inputs to always have the same statistical properties (mean of 0 and standard deviation of 1) before passing them to the next layer. Even when Layer 1's weights change dramatically, the normalization ensures Layer 2 always receives inputs with consistent statistics. This means Layer 2 can focus on learning the right transformations instead of constantly adapting to different input scales. The network can learn stable patterns, train faster, and use higher learning rates without becoming unstable. It's like giving each layer a stable platform to stand on, rather than making them balance on a constantly shifting surface.

### Why it doesnt affects the message that the weights wanted to convey through high or low magnitude

The key insight is that **layer norm doesn't destroy the information** - it transforms it from being encoded in absolute magnitudes to being encoded in **relative patterns** across features.

Here's why the "message" is preserved:

**1. Relative relationships matter, not absolute scale** When you normalize a vector like [10, 2, 5] → it becomes something like [1.2, -0.9, 0.3]. The relative ordering and proportions are preserved - you can still see which dimensions were larger/smaller relative to each other. The weights learn to encode information in these _patterns_ rather than raw magnitudes.

**2. Learned affine parameters (γ and β) restore expressiveness** After normalizing to mean=0, variance=1, layer norm applies: `γ * normalized_x + β`

These learned parameters γ (scale) and β (shift) are **per-feature**, meaning each dimension can be rescaled independently. So if a particular feature dimension is important, the network can learn a large γ for that dimension to amplify it again after normalization.

**3.What actually changes:** The weights can't use "this activation is 1000 vs 0.1" to convey meaning _within a single layer's output_. Instead they convey meaning through "feature A is 3x larger than feature B" and through the learned γ/β rescaling.

### Difference between Layer Norm and Using Sqrt of d_k in scaled dot product attention

We use sqrt of d_k in scaled dot product attention so that we can reduce the magnitude of the difference between the vectors, how similar or different is that idea is from Layer Norm? Because in Layer Norm as well we are just normalizing the numbers right? So whats the difference?

**The key mathematical difference:**

**÷√d_k preserves absolute relationships:**

- [10, 20, 30] → [10/√d_k, 20/√d_k, 30/√d_k]
- If you multiply everything by the same constant, relative AND absolute differences are preserved
- The vector [10, 20, 30] stays proportional to itself

**Layer norm destroys absolute scale but preserves relative patterns:**

- [10, 20, 30] → normalize to something like [-1.22, 0, 1.22]
- [100, 200, 300] → also normalizes to [-1.22, 0, 1.22] (same result!)
- The absolute magnitude information is **lost** - you can't tell if the original values were in the 10s or 1000s
- Only the relative pattern (this was smallest, this was middle, this was largest) is preserved

**if the output of different weights after norm is same , that doesnt sound right**
Layer norm makes vectors with the _same relative pattern_ but different magnitudes produce the same output. But in practice, **weight operations change the patterns, not just the scale.**

**What actually happens in a transformer:**

```
# Token 1 path:
x1 = [1, 2, 3]
After weights W: Wx1 = [15, 30, 22]  # Pattern changed by W
After layer norm: normalized based on [15, 30, 22]

# Token 2 path:  
x2 = [10, 20, 30]  # Same pattern, different scale
After weights W: Wx2 = [150, 300, 220]  # Different scale, same pattern
After layer norm: Gets same normalized output as token 1!
```

While Scaled Dot-Product Attention and Layer Normalization both utilize division to control magnitude, they are functionally opposites: Attention scales passively to **preserve** signal confidence, whereas Layer Normalization scales aggressively to **destroy** signal intensity. Attention divides by a static constant ($\sqrt{d_k}$) purely to prevent the Softmax function from saturating, ensuring that the relative difference between a "strong" match and a "weak" match remains intact so the model retains a sense of certainty. 

In contrast, Layer Normalization divides by the input's own standard deviation, forcing every vector onto the exact same statistical scale (unit variance) regardless of its original strength. This process renders the layer "scale invariant"—meaning it treats a quiet signal (e.g., "Happy") and a loud signal (e.g., "Very Happy") as mathematically identical—deliberately sacrificing the model's ability to express intensity through magnitude (radial expressivity) in exchange for a stable, spherical optimization landscape where gradients are normalized and training does not diverge.

Layer norm doesnt care about magnitude if there are two different vectors like these 
Vector A - [1,2]
Vector B - [100,102]

then to layer norm they both can compress down to [-1,+1], they dont provide different "pattern" to the system and hence nothing new for the model according to layer norm, but this is very different in case of scaled dot product attention, in that component magnitude is everything, we want to know what is more intense and what is not so that softmax can amplify it further, we just perform normlization there because otherwise softmax's exp would blow up the computer.

I THOUGHT THAT BECAUSE BOTH ARE JUST NORMALIZING AND REDUCING THE MAGNITUDE WHY NOT JUST USE LAYER NORM IN SCALED DOT PRODUCT AS WELL BUT I NOW KNOW WHY , THEY SOLVE VERY DIFFERENT PROBLEMS AND INTERCHANGING THEM WILL JUST KILL BOTH FRONTS

### why layer norm is a learnable thing? isnt it just math which brings std to 1?
Layer norm does normalize the input (mean 0, std 1), but if that were all it did, it might erase useful scale or shift information the model needs. That’s why it has learnable parameters: γ (scale) and β (shift). They let the network restore or adjust the normalized representation if the raw scale or offset actually matters for learning. So the normalization stabilizes training, but the learnable part ensures the model doesn’t lose the freedom to represent what it needs.

### What would happen if LN is not learnable?

LayerNorm doesn’t need a regulator because its learnable parameters act only on already-normalized activations, and normalization is applied every time before those parameters are used, which prevents runaway scale.

### Why layer norm learnable parameters just doesnt explodes

**Layer normalization is supposed to keep activations stable during training, but layer norm itself has learnable parameters (γ and β). If these parameters are updated via the same gradient descent process that causes instability in regular weights, what prevents the layer norm parameters themselves from becoming unstable?**

The key point is that LayerNorm’s stabilizing effect comes primarily from the normalization step itself, not from γ and β. Even though γ (scale) and β (shift) are learnable, they’re applied after activations are normalized to zero mean and unit variance, so their updates are operating on a well-conditioned signal. That already removes most of the sources of exploding or vanishing behavior. In practice, γ and β tend to change slowly and stay well-behaved, especially compared to raw weights. If they did drift too much, the normalization in the next forward pass would still re-center and re-scale activations, which acts as a built-in corrective mechanism.



