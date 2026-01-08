## Bumblebee — A Toy Transformer From Scratch

Bumblebee is a **minimal Transformer implementation written from scratch**.
The goal of this project is to understand how Transformers work at both the
theoretical and implementation level, by rebuilding the core components
without relying on high-level abstractions.

The implementation focuses on core building blocks (embeddings, attention, residuals, normalization) rather than training or large-scale optimization

---

### Primary learning resources

These are the main external resources that helped shape this implementation:

1. [Umar Jamil — Transformer Explanation ](https://www.youtube.com/watch?v=bCz4OMemCcA&t=484s) 
2. [Umar Jamil — Transformer Implementation ](https://www.youtube.com/watch?v=ISNdQcPhsts&t=389s)
3. [The Annotated Transformer (Harvard NLP) ](https://nlp.seas.harvard.edu/annotated-transformer/) 
4. [An Even More Annotated Transformer (pi-tau) ](https://pi-tau.github.io/posts/transformer/#attention) 
5. [UVA Deep Learning Notebooks — Transformers & Multi-Head Attention  ](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)
6. [Naoki Shibuya — Positional Encoding Explained](https://naokishibuya.github.io/blog/2021-10-31-transformers-positional-encoding/)  
7. [Hugging Face — Designing Positional Encoding](https://huggingface.co/blog/designing-positional-encoding](https://huggingface.co/blog/designing-positional-encoding)  
8. [Towards Data Science — Deriving Positional Encoding](https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3](https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3)  
9. [Timo Denk — Linear Relationships in Positional Encoding ](https://blog.timodenk.com/linear-relationships-in-the-transformers-positional-encoding/) 
10. [AI StackExchange — Why √dₖ is used in Scaled Dot-Product Attention  ](https://ai.stackexchange.com/questions/41861/why-use-a-square-root-in-the-scaled-dot-product)

---

### Implementation notes & personal walkthroughs

While building this project, I wrote a few short posts to clarify specific
implementation details and intuitions:

- [Why LayerNorm and √dₖ Scaling Are Not the Same Thing ](https://www.linkedin.com/feed/update/urn:li:activity:7409122355836313600/)
- [Scaling in Scaled Dot-Product Attention ](https://dev.to/samyak112/scaling-is-all-you-need-understanding-sqrtd-in-self-attention-29pk) 
- [How Linear Projections Enable Head-Wise Specialization](https://www.linkedin.com/feed/update/urn:li:activity:7411658892079820800/) 
  

These notes reflect my own understanding while implementing the model and are
informed by discussions in research-focused communities such as Hugging Face
and Yannic Kilcher’s Discord.
