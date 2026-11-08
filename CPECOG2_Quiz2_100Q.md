# CPECOG2 Quiz 2: 100-Item Exam

**Coverage:** Recurrent Neural Networks · LSTM, GRU and SRU · Challenges and Solutions in Training RNNs · Transfer Learning in CNNs

| Part | Type | Items |
|---|---|---|
| I | True or False | 1–30 |
| II | Multiple Choice (includes computation) | 31–80 |
| III | Matching Type | 81–100 |

The answer key is at the end of the file.

---

## Part I. True or False (Items 1–30)

Write **TRUE** if the statement is correct. Write **FALSE** if it is not.

1. A vanilla RNN applies the same weights at every time step.
2. In a vanilla RNN, the hidden state hₜ depends only on the current input xₜ.
3. The number of parameters of a vanilla RNN grows as the sequence length T grows.
4. BPTT is ordinary backpropagation applied to the unrolled computational graph.
5. Because weights are shared across time, the gradient of a shared weight in BPTT is the average of its contributions from all time steps.
6. A many-to-one architecture is a natural fit for sentiment classification.
7. Bidirectional RNNs are well suited to real-time streaming prediction.
8. Perplexity is the exponential of the mean cross-entropy.
9. If the largest singular value of each recurrent Jacobian factor is below 1, gradients tend to vanish over long spans.
10. Gradient clipping is the standard fix for vanishing gradients.
11. Gradient clipping rescales an oversized gradient and preserves its direction.
12. Truncated BPTT reduces memory from O(T) to O(k), where k is the chunk length.
13. The output range of the tanh activation is (0, 1).
14. In the BPTT backward recursion, δ(T+1) is initialized to zero.
15. RNN training can be fully parallelized across time steps, just as convolutions are parallelized across image positions.
16. The LSTM cell-state update is additive: cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ.
17. A GRU has three gates, including an output gate.
18. Initializing the LSTM forget-gate bias to about 1 is a common practice.
19. SRU gates do not use a matrix multiply on the previous hidden state, which allows most of the computation to run in parallel across time.
20. For the same D and H, a GRU has fewer parameters than an LSTM.
21. Greff et al. (2017) found one LSTM variant that clearly outperformed the standard LSTM.
22. Randomly shuffling a time series before splitting it into train and test sets is an acceptable evaluation practice.
23. Dropout and early stopping are both used to reduce overfitting.
24. When sequences are padded to equal length, the padded steps should still contribute to the loss.
25. In a pretrained CNN, early layers tend to be more general than late layers.
26. Training from scratch is the best choice when the target dataset is very small.
27. In feature extraction, the pretrained backbone weights are updated during training.
28. In fine-tuning, the pretrained layers are usually trained at a smaller learning rate than the new head.
29. BatchNorm layers should be kept in training mode when fine-tuning on small batches.
30. Negative transfer happens when pretrained weights perform worse than training from scratch.

---

## Part II. Multiple Choice (Items 31–80)

Choose the letter of the best answer. For computation items, show your solution on scratch paper.

**31.** Which statement best describes the hidden state of an RNN?
- A. A lookup table of word embeddings
- B. The set of training labels
- C. A vector summarizing the sequence seen so far, updated at each step
- D. The final output probability

**32.** Which sequence pattern fits image captioning (one image in, a word sequence out)?
- A. One-to-many
- B. Many-to-one
- C. One-to-one
- D. Many-to-many with equal lengths

**33.** Which setup is used for translation, where input and output lengths differ?
- A. Many-to-one
- B. One-to-many
- C. A single-layer unidirectional RNN
- D. Encoder-decoder

**34.** Which activation is standard in the vanilla RNN cell?
- A. ReLU
- B. tanh
- C. Sigmoid
- D. Softmax

**35.** What is the main benefit of parameter sharing across time steps?
- A. The parameter count does not depend on sequence length, and the same pattern is learned at any position
- B. It removes the need for an activation function
- C. It guarantees gradients never vanish
- D. It lets each step use a different weight matrix

**36.** BPTT memory per sequence grows with sequence length T in which way?
- A. Constantly
- B. Logarithmically
- C. Linearly
- D. Exponentially

**37.** Which technique keeps padded time steps out of the loss?
- A. Dropout
- B. Masking
- C. Gradient clipping
- D. Weight decay

**38.** Bidirectional layers are unsuitable for which kind of task?
- A. Sentiment classification
- B. Named entity tagging
- C. Speech recognition with the full recording available
- D. Streaming or forecasting, where future inputs are unavailable

**39.** *(Computation)* A vanilla RNN has D = 100 inputs, H = 64 hidden units and C = 5 outputs. How many parameters does it have (Wxh, Whh, bh, Why and by)?
- A. 10,885
- B. 10,560
- C. 10,820
- D. 6,400

**40.** *(Computation)* A dense layer flattens T = 50 steps of D = 20 features into H = 64 units. How many weights does it need (no biases)?
- A. 1,280
- B. 5,376
- C. 64,000
- D. 69,376

**41.** *(Computation)* By what factor does the number of parameters in Whh grow when H doubles from 128 to 256?
- A. ×2
- B. ×4
- C. ×8
- D. ×16

**42.** *(Computation)* Scalar RNN: hₜ = tanh(w·xₜ + u·hₜ₋₁) with w = 0.5, u = 0.8, h₀ = 0 and x₁ = 1.0. What is h₁?
- A. 0.500
- B. 0.400
- C. 0.693
- D. 0.462

**43.** *(Computation)* Using the same RNN with h₁ = 0.462 and x₂ = 0.5, what is h₂?
- A. 0.551
- B. 0.462
- C. 0.690
- D. 0.250

**44.** *(Computation)* A softmax output is p = (0.7, 0.2, 0.1) and the true class is class 1 (one-hot target y = (1, 0, 0)). What is the gradient p − y of the cross-entropy loss with respect to the logits?
- A. (0.7, 0.2, 0.1)
- B. (−0.7, 0.2, 0.1)
- C. (−0.3, 0.2, 0.1)
- D. (0.3, −0.2, −0.1)

**45.** *(Computation)* A language model has a mean cross-entropy of 2.0 nats. What is its perplexity?
- A. 2.00
- B. 7.39
- C. 4.00
- D. 20.09

**46.** Which LSTM gate scales the old cell state, so a value near 0 erases it?
- A. Input gate
- B. Output gate
- C. Reset gate
- D. Forget gate

**47.** Which equation is the LSTM cell-state update?
- A. cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ
- B. cₜ = tanh(W·xₜ + U·cₜ₋₁)
- C. cₜ = oₜ ⊙ tanh(hₜ)
- D. cₜ = (1 − zₜ) ⊙ cₜ₋₁

**48.** Why does the LSTM mitigate vanishing gradients?
- A. It uses a larger learning rate
- B. It clips every gradient
- C. The additive cell path gives ∂cₜ/∂cₜ₋₁ = fₜ, so the signal is not repeatedly squashed
- D. It removes the recurrent connection

**49.** The LSTM forget-gate bias is commonly initialized to about:
- A. −1
- B. 1
- C. 0.01
- D. 100

**50.** *(Computation)* How many parameters does an LSTM layer have with D = H = 64, counting biases and using 4H(D + H + 1)?
- A. 8,256
- B. 24,768
- C. 16,512
- D. 33,024

**51.** *(Computation)* How many parameters does a GRU layer have with D = H = 64, using 3H(D + H + 1)?
- A. 24,768
- B. 33,024
- C. 8,256
- D. 16,512

**52.** *(Computation)* How many parameters does an SRU layer have with D = H = 64, using 3HD + 4H?
- A. 12,288
- B. 8,256
- C. 12,544
- D. 24,768

**53.** *(Computation)* How many parameters does an LSTM layer have with D = 32 and H = 64, counting biases?
- A. 33,024
- B. 24,832
- C. 24,576
- D. 12,416

**54.** *(Computation)* For equal D and H, a GRU has what fraction of an LSTM’s parameters?
- A. 25%
- B. 50%
- C. 33%
- D. 75%

**55.** Which cell targets speed instead of memory?
- A. SRU
- B. LSTM
- C. GRU
- D. Elman RNN

**56.** Why is the SRU faster than the LSTM?
- A. It has more gates
- B. It uses a bigger hidden size
- C. Its gates do not need a matrix multiply on the previous state, so they can be computed for all time steps in parallel
- D. It removes the input weights

**57.** Which two gates does the GRU use?
- A. Input and output
- B. Update and reset
- C. Forget and output
- D. Input and forget

**58.** What did Greff et al. (2017) conclude about LSTM variants?
- A. Removing the output gate always helps
- B. Peephole connections are required
- C. GRU always beats LSTM
- D. No variant was clearly better than the standard LSTM

**59.** Which cell is a reasonable choice for streaming or edge devices?
- A. A deep stack of bidirectional LSTMs
- B. A large Transformer
- C. A small GRU or LSTM
- D. A very wide vanilla RNN

**60.** Vanishing gradients occur when the Jacobian factors in the BPTT product generally have norms that are:
- A. Greater than 1
- B. Less than 1
- C. Exactly 1
- D. Negative

**61.** *(Computation)* If each step keeps 90% of the signal, about how much remains after 20 steps (0.9²⁰)?
- A. 12%
- B. 18%
- C. 2%
- D. 36%

**62.** *(Computation)* If each step grows the signal by 10%, about how many times larger is it after 30 steps (1.1³⁰)?
- A. 3
- B. 30
- C. 17
- D. 33

**63.** *(Computation)* If each step keeps 90% of the signal, about how much remains after 30 steps (0.9³⁰)?
- A. 27%
- B. 4%
- C. 12%
- D. 0.4%

**64.** *(Computation)* The gradient is g = (6, 8) and the clipping threshold is τ = 5. After norm clipping, what is g?
- A. (6, 8)
- B. (5, 5)
- C. (0.6, 0.8)
- D. (3, 4)

**65.** *(Computation)* A sequence of T = 1,000 steps is trained with truncated BPTT using k = 50. How many chunks per sequence?
- A. 20
- B. 50
- C. 1,000
- D. 5

**66.** The loss suddenly spikes or turns into NaN. Which remedy applies?
- A. Early stopping
- B. Dropout
- C. Gradient clipping
- D. Padding

**67.** What is the main trade-off of truncated BPTT?
- A. Higher memory use
- B. Dependencies longer than the chunk length may be missed
- C. Slower per-step computation
- D. It requires bidirectional layers

**68.** Which practice prevents leakage when evaluating a forecasting model?
- A. Random shuffling
- B. Random k-fold cross-validation
- C. Training on the test set
- D. A chronological (time-ordered) split

**69.** Which of the following is NOT a remedy for overfitting?
- A. Dropout
- B. Early stopping
- C. Training for many more epochs without monitoring validation loss
- D. Getting more data

**70.** Exposure bias from teacher forcing can be reduced by:
- A. Larger batch size
- B. Scheduled sampling
- C. Weight decay
- D. Bidirectional layers

**71.** Which description of ImageNet-1k is correct?
- A. About 10,000 images and 10 classes
- B. About 50,000 images and 100 classes
- C. About 1.28 million images and 10 classes
- D. About 1.28 million images and 1,000 classes

**72.** Early layers of a pretrained CNN mainly detect:
- A. Edges, colors and simple textures
- B. Whole objects
- C. Class labels
- D. Sentence structure

**73.** Which strategy suits a small target dataset whose images resemble ImageNet?
- A. Training from scratch
- B. Fine-tuning every layer at a high learning rate
- C. Feature extraction with a frozen backbone and a new head
- D. Using random weights

**74.** Why warm up the new head before unfreezing backbone layers?
- A. To increase the image size
- B. A random head produces large gradients that can damage pretrained weights
- C. To remove BatchNorm
- D. To avoid using labels

**75.** *(Computation)* A new ResNet-50 head maps 2,048 features to C = 10 classes (weights plus biases). How many parameters is that?
- A. 20,480
- B. 2,058
- C. 10,245
- D. 20,490

**76.** *(Computation)* ResNet-50’s original fc layer has 2,049,000 parameters out of about 25.6 million. About what share is that?
- A. 8%
- B. 80%
- C. 0.8%
- D. 40%

**77.** Which learning-rate setup is typical when fine-tuning?
- A. The same large rate everywhere
- B. A larger rate for the backbone than the head
- C. A smaller rate for the pretrained backbone than for the new head
- D. A learning rate of exactly zero for the head

**78.** In PyTorch, how do you freeze a parameter p?
- A. p.trainable = False
- B. p.requires_grad = False
- C. p.freeze = True
- D. p.detach = True

**79.** When fine-tuning on small batches, BatchNorm layers should usually be:
- A. Replaced by dropout
- B. Reinitialized randomly
- C. Doubled in number
- D. Kept in inference mode, using the stored statistics

**80.** For a medical-imaging dataset with several images per patient, the data should be split by:
- A. Patient
- B. Image
- C. File name
- D. Image size

---

## Part III. Matching Type (Items 81–100)

### Set A (Items 81–90): Match each term to its description

Write the letter of the correct description. Each letter is used at most once. Two descriptions are extras.

| No. | Term |
|---|---|
| 81 | Elman (vanilla) RNN |
| 82 | Truncated BPTT |
| 83 | Gradient clipping |
| 84 | LSTM |
| 85 | GRU |
| 86 | SRU |
| 87 | Feature extraction |
| 88 | Fine-tuning |
| 89 | Perplexity |
| 90 | Linear probe |

| Letter | Description |
|---|---|
| A | Two-gate cell (update, reset) with a single state hₜ |
| B | Rescales an oversized gradient while keeping its direction |
| C | Exponential of the mean cross-entropy; lower is better |
| D | Randomly switches off units during training |
| E | Freeze the pretrained backbone and train only a new head |
| F | Vanilla recurrent network: hₜ = tanh(Wxh xₜ + Whh hₜ₋₁ + bh) |
| G | Cell whose gates avoid a matrix multiply on the previous state, for speed |
| H | Train a linear classifier on frozen features to measure their quality |
| I | Backpropagate only k steps within each chunk |
| J | Gated cell with a separate cell state cₜ and input, forget and output gates |
| K | Skip connections that let a signal bypass steps |
| L | Continue training pretrained layers at a small learning rate |

### Set B (Items 91–100): Match each problem to its remedy

Write the letter of the best remedy. Each letter is used at most once. Two remedies are extras.

| No. | Problem or symptom |
|---|---|
| 91 | Loss spikes or turns into NaN |
| 92 | Early steps barely learn from errors made much later (fading signal) |
| 93 | Training score keeps improving while validation score gets worse |
| 94 | A time-series model scores suspiciously well after random shuffling |
| 95 | Sequences in a batch have different lengths |
| 96 | Mismatch between teacher-forced training and free-running test |
| 97 | Tiny target dataset with ImageNet-like images |
| 98 | A random new head damages pretrained weights early in training |
| 99 | BatchNorm statistics are unstable with small batches |
| 100 | Grayscale images fed to an RGB-pretrained CNN |

| Letter | Remedy |
|---|---|
| A | Keep BatchNorm layers in inference mode |
| B | Feature extraction with a frozen backbone |
| C | Chronological (time-ordered) split |
| D | Increase the learning rate |
| E | Scheduled sampling |
| F | Replicate the single channel to 3 channels |
| G | Gated cells (LSTM or GRU) |
| H | Gradient clipping |
| I | Bidirectional layer |
| J | Padding with masking |
| K | Warm up the new head before unfreezing backbone layers |
| L | Dropout and early stopping |

---

# Answer Key

## Part I. True or False

| No. | Ans. | No. | Ans. | No. | Ans. |
|---|---|---|---|---|---|
| 1 | TRUE | 11 | TRUE | 21 | FALSE |
| 2 | FALSE | 12 | TRUE | 22 | FALSE |
| 3 | FALSE | 13 | FALSE | 23 | TRUE |
| 4 | TRUE | 14 | TRUE | 24 | FALSE |
| 5 | FALSE | 15 | FALSE | 25 | TRUE |
| 6 | TRUE | 16 | TRUE | 26 | FALSE |
| 7 | FALSE | 17 | FALSE | 27 | FALSE |
| 8 | TRUE | 18 | TRUE | 28 | TRUE |
| 9 | TRUE | 19 | TRUE | 29 | FALSE |
| 10 | FALSE | 20 | TRUE | 30 | TRUE |

**Corrections for the FALSE items**

- **2.** hₜ depends on both xₜ and hₜ₋₁.
- **3.** The parameter count does not depend on T, because the weights are shared.
- **5.** The gradient of a shared weight is the **sum** of its contributions, not the average.
- **7.** Bidirectional layers need the whole sequence, including future inputs, so they do not suit streaming or forecasting.
- **10.** Clipping addresses **exploding** gradients. Vanishing gradients call for gated cells, skip connections and similar fixes.
- **13.** The range of tanh is (−1, 1).
- **15.** Step t needs step t−1, which limits parallelism across time.
- **17.** A GRU has two gates (update and reset) and no output gate.
- **21.** Greff et al. found no variant clearly better than the standard LSTM.
- **22.** Use a chronological split. Random shuffling leaks the future into training.
- **24.** Masking removes padded steps from the loss.
- **26.** With a tiny target dataset, use a pretrained model (such as feature extraction), not training from scratch.
- **27.** The backbone is frozen. Only the new head is trained.
- **29.** Keep BatchNorm layers in inference mode so the stored statistics are used.

## Part II. Multiple Choice

| No. | Ans. | No. | Ans. | No. | Ans. | No. | Ans. | No. | Ans. |
|---|---|---|---|---|---|---|---|---|---|
| 31 | C | 41 | B | 51 | A | 61 | A | 71 | D |
| 32 | A | 42 | D | 52 | C | 62 | C | 72 | A |
| 33 | D | 43 | A | 53 | B | 63 | B | 73 | C |
| 34 | B | 44 | C | 54 | D | 64 | D | 74 | B |
| 35 | A | 45 | B | 55 | A | 65 | A | 75 | D |
| 36 | C | 46 | D | 56 | C | 66 | C | 76 | A |
| 37 | B | 47 | A | 57 | B | 67 | B | 77 | C |
| 38 | D | 48 | C | 58 | D | 68 | D | 78 | B |
| 39 | A | 49 | B | 59 | C | 69 | C | 79 | D |
| 40 | C | 50 | D | 60 | B | 70 | B | 80 | A |

**Worked solutions for the computation items**

- **39.** Wxh = 64 × 100 = 6,400. Whh = 64 × 64 = 4,096. bh = 64. Why = 5 × 64 = 320. by = 5. Total = **10,885**.
- **40.** 50 × 20 × 64 = **64,000** weights. For comparison, the RNN’s Wxh + Whh = 64·20 + 64·64 = 5,376.
- **41.** Whh has H² entries. 256² / 128² = **4**.
- **42.** h₁ = tanh(0.5 × 1.0 + 0.8 × 0) = tanh(0.5) = **0.462**.
- **43.** h₂ = tanh(0.5 × 0.5 + 0.8 × 0.462) = tanh(0.6196) = **0.551**.
- **44.** p − y = (0.7 − 1, 0.2 − 0, 0.1 − 0) = **(−0.3, 0.2, 0.1)**.
- **45.** Perplexity = exp(2.0) ≈ **7.39**.
- **50.** 4H(D + H + 1) = 4 × 64 × 129 = **33,024**.
- **51.** 3H(D + H + 1) = 3 × 64 × 129 = **24,768**.
- **52.** 3HD + 4H = 3 × 64 × 64 + 4 × 64 = 12,288 + 256 = **12,544**.
- **53.** 4 × 64 × (32 + 64 + 1) = 4 × 64 × 97 = **24,832**.
- **54.** 3H(D + H + 1) / 4H(D + H + 1) = 3/4 = **75%**.
- **61.** 0.9²⁰ ≈ 0.122, so about **12%**.
- **62.** 1.1³⁰ ≈ 17.4, so about **17×**.
- **63.** 0.9³⁰ ≈ 0.042, so about **4%**.
- **64.** ‖g‖ = √(6² + 8²) = 10 > 5, so scale by τ/‖g‖ = 0.5. g becomes **(3, 4)**.
- **65.** 1,000 / 50 = **20** chunks.
- **75.** 2,048 × 10 + 10 = **20,490**.
- **76.** 2,049,000 / 25,600,000 ≈ 0.080, so about **8%**.

## Part III. Matching Type

**Set A**

| No. | Ans. | No. | Ans. |
|---|---|---|---|
| 81 | F | 86 | G |
| 82 | I | 87 | E |
| 83 | B | 88 | L |
| 84 | J | 89 | C |
| 85 | A | 90 | H |

Extras (not used): **D** (dropout) and **K** (skip connections).

**Set B**

| No. | Ans. | No. | Ans. |
|---|---|---|---|
| 91 | H | 96 | E |
| 92 | G | 97 | B |
| 93 | L | 98 | K |
| 94 | C | 99 | A |
| 95 | J | 100 | F |

Extras (not used): **D** (increase the learning rate) and **I** (bidirectional layer).
