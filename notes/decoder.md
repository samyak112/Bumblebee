### My Understanding of decoder block

The Decoder is the generative engine of the Transformer, designed to overcome the fundamental bottleneck of older **RNN architectures**. In an RNN, the entire history of a sentence had to be compressed into a single, fixed-size hidden state, which quickly became "bloated" and "muddy" as sequences grew longer, causing the model to forget early details.

The Transformer Decoder removes recurrent hidden state by recomputing representations from scratch (or from cached keys/values during inference). Instead of passing a single degrading vector forward, it effectively "re-reads" and **re-embeds the entire "so far" generated sequence from scratch** at every single iteration.

The "so far" comes from the fact that the Transformer Decoder operates in a **loop** during generation (inference). This is called **Autoregressive Generation**.

Unlike the Encoder, which processes the entire input sentence at once (in parallel), the Decoder is forced to generate the output **one word at a time**.1

Basically there are 3 steps in the decoder: masked MHA, cross attention, and then the good old feed forward.

We have two MHAs here because we need to do two different jobs. The first MHA is masked—why? Because the model is autoregressive. It generates the next token based on whatever's already generated. By that logic, if we passed one token at a time it would take forever. So instead, we pass the whole sentence but make the model see limited content at each position. Like if the sentence is "Hi how are you":

- First position only sees "Hi" so it can predict "how"
- Next sees "Hi how" so it can predict "are"
- Then "Hi how are" so it can predict the next token

Without masking decoder MHA would get

$$\text{Scores} = \begin{bmatrix} (\text{The} \cdot \text{The}) & (\text{The} \cdot \text{black}) & (\text{The} \cdot \text{cat}) \\ (\text{black} \cdot \text{The}) & (\text{black} \cdot \text{black}) & (\text{black} \cdot \text{cat}) \\ (\text{cat} \cdot \text{The}) & (\text{cat} \cdot \text{black}) & (\text{cat} \cdot \text{cat}) \end{bmatrix}$$

but instead with mask it gets 

$$\text{Masked Scores} = \begin{bmatrix} (\text{The} \cdot \text{The}) & \color{red}{-\infty} & \color{red}{-\infty} \\ (\text{black} \cdot \text{The}) & (\text{black} \cdot \text{black}) & \color{red}{-\infty} \\ (\text{cat} \cdot \text{The}) & (\text{cat} \cdot \text{black}) & (\text{cat} \cdot \text{cat}) \end{bmatrix}$$

Now each row only sees enough that it should be able to because of negative infinity

All these positions are processed in parallel, so it's still autoregressive, but since we already have the whole sentence, we can cheat for faster training.

Now why do we do this? We want to contextualize the embedding at each step. Let's say the sentence "hi how" has been generated—before generating the next word, we contextualize the whole thing. We're basically saying "hey, look at your current state and understand your surroundings (just not the future)."

But then how does this contextualized embedding become the Query for cross attention?

We pass the **contextualized decoder representation** into the next attention block, project it with a **query matrix**, and **training pressure shapes that representation so it becomes a _good query_**.

Now comes cross attention. Why do we need this? Because so far, the decoder only knows about itself. It knows the grammar—like "I just said 'The black', so I probably need a noun next"—but it has no clue _which_ noun because it hasn't looked at the source sentence yet.

So we use that refreshed, contextualized embedding from the previous step as the **Query**. This Query goes to the encoder output (which acts as the **Keys and Values**) and basically asks: "Which part of the source sentence is relevant to what I'm trying to say right now?"

Since there's no mask here, the decoder can look at the entire source sentence at once. It finds the best match—like finding the vector for 'chat' in the source when it needs a noun—and pulls that information (the Value) into the decoder stream.

Now our embedding has both the grammatical context (from the first MHA) and the actual translation content (from the second MHA).

Finally, we hit the feed forward network. This part is simple. We just passed the vector through two heavy attention layers, mixing a lot of info. The FFN is just a couple of linear layers with a ReLU that processes this new "hybrid" vector, shaping the features one last time so the final output layer can easily pick the right word from the dictionary.


### How does negative infinity helps

It helps us by exploiting the mathematical property of the **Softmax** function to physically break the connection between words.

If you don't use Negative Infinity, you cannot "hide" a word.

Here is the math of **why** we need such an extreme number.

### 1. The Softmax Formula

Recall that Attention Scores (logits) are passed through the Softmax function to turn them into probabilities (weights):1

$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$$

The key term is 2**$e^x$** (Euler's number to the power of the score).3


### 2. Why we can't use Zero

You might think: _"If I want to mask a word, why don't I just set its score to 0?"_

Let's see what happens if we set the score for "sat" (the future word) to 0:

$$e^0 = 1$$

If you have 4 words and you mask the last one with 0, the exponential values might look like:

[2.7, 5.2, 1.8, 1.0] (The 1.0 is the "masked" word).

When you normalize this, that 1.0 will result in a probability of maybe 0.1 (10%).

Result: The model still peeks at the future word with 10% attention. The mask failed.

### 3. Why we use Negative Infinity

Now, let's set the score to $-\infty$ (or -1e9 in code):

$$e^{-\infty} = \frac{1}{e^{\infty}} = 0$$

If you have 4 words and mask the last one with -1e9:

[2.7, 5.2, 1.8, 0.0000000...]

When you normalize this, the probability for that word is exactly 0%.

Result:

1. **Value Contribution:** $0 \times \text{ValueVector} = 0$. (No information flows).
2. **Gradient:** The gradient for that position is 0. (The model doesn't learn from it).


Negative Infinity is the **only** way to get a pure **Zero** out of a Softmax function.

- **Mask with 0** $\rightarrow$ Softmax outputs positive number (Leakage).
- **Mask with $-\infty$** $\rightarrow$ Softmax outputs 0 (Perfect Seal).