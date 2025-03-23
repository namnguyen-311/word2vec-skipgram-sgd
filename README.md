# word2vec-skipgram-sgd
Manual implementation of Skip-Gram Word2Vec with Negative Sampling and Stochastic Gradient Descent from Stanford CS224N Lecture 2.

# Word2Vec Skip-Gram (Manual Implementation)

This is a simple Python implementation of **one training step** of the **Skip-Gram Word2Vec model with Negative Sampling**, inspired by Stanford's CS224N (Lecture 2).

## What It Does
- Computes dot products between word vectors
- Applies sigmoid function
- Calculates the loss using negative sampling
- Computes gradient w.r.t the center word vector
- Updates the vector using Stochastic Gradient Descent (SGD)

## Concepts Covered
- Word Embeddings
- Skip-Gram Model
- Negative Sampling
- SGD (Stochastic Gradient Descent)
- Dot Product & Sigmoid
- Loss Function Optimization

## Output Example
Loss: 2.021177
Gradient: [-0.094201  0.008991  0.084183]
Updated v_c: [0.10471005 0.19955045 0.29579085]

## Why It Matters
This project demonstrates core principles of word vector training **from scratch**, without relying on high-level libraries. It’s great for anyone trying to deeply understand how models like Word2Vec learn embeddings.

---

> Built by Nam Nguyen as part of CS224N hands-on learning.
