# CPECOG2 Quiz 2: Term–Definition Matching

**Coverage:** Recurrent Neural Networks · LSTM, GRU and SRU · Challenges and Solutions in Training RNNs · Transfer Learning in CNNs

**Directions:** For each set, match every term (Column A) to its definition (Column B). Write the letter only. Each letter is used at most once, and **two definitions in every set are extras** that match no term.

| Set | Topic | Items |
|---|---|---|
| 1 | RNN fundamentals | 1–10 |
| 2 | LSTM, GRU and SRU | 11–20 |
| 3 | Training challenges and remedies | 21–30 |
| 4 | Transfer learning in CNNs | 31–40 |
| 5 | Implementation and practice | 41–50 |

---

## Set 1. RNN Fundamentals (Items 1–10)

| No. | Column A: Term |
|---|---|
| 1 | Hidden state |
| 2 | Parameter sharing |
| 3 | Unrolling |
| 4 | Backpropagation through time (BPTT) |
| 5 | Many-to-one |
| 6 | One-to-many |
| 7 | Encoder-decoder |
| 8 | Bidirectional RNN |
| 9 | Perplexity |
| 10 | Masking |

| Letter | Column B: Definition |
|---|---|
| A | A single input produces a sequence of outputs, as in image captioning |
| B | Backpropagation applied to the unrolled graph, summing the gradient contributions of the tied weights over all steps |
| C | Expanding the recurrence across time into a graph with one copy of the cell per step |
| D | Reusing the same weights at every time step |
| E | Excluding padded time steps from the loss |
| F | Downsamples a feature map by taking the maximum over local windows |
| G | A vector summarizing the sequence seen so far, updated at each step |
| H | One network reads the input sequence and another generates the output sequence, whose length may differ |
| I | A whole sequence goes in and a single output comes out, as in sentiment classification |
| J | The exponential of the mean cross-entropy, the usual language-model metric |
| K | The step size used in each gradient-descent update |
| L | Runs one RNN forward and another backward over the sequence and combines their states |

---

## Set 2. LSTM, GRU and SRU (Items 11–20)

| No. | Column A: Term |
|---|---|
| 11 | Cell state (cₜ) |
| 12 | Forget gate |
| 13 | Input gate |
| 14 | Output gate |
| 15 | Update gate (GRU) |
| 16 | Reset gate (GRU) |
| 17 | Additive cell update |
| 18 | SRU |
| 19 | Highway term (SRU) |
| 20 | Forget-gate bias initialization |

| Letter | Column B: Definition |
|---|---|
| A | Scales the old cell state: a value near 1 keeps it and a value near 0 erases it |
| B | A gated cell whose gates avoid a matrix multiply on the previous state, so most of its work runs in parallel across time |
| C | Normalizes activations to zero mean and unit variance within each mini-batch |
| D | Controls how much of the cell is exposed as the hidden state hₜ |
| E | Interpolates between keeping the old state and writing the candidate |
| F | The term (1 − rₜ) ⊙ xₜ, which lets the input pass straight to the output |
| G | Setting the bias to about 1 so the cell remembers information early in training |
| H | Decides how much of the candidate content gₜ is written into the cell |
| I | The LSTM’s protected memory line, read out through the output gate |
| J | Controls how much of the previous state is used to form the candidate |
| K | A convolution that pads its input so the output keeps the same spatial size |
| L | cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ, which gives gradients a direct path through time when fₜ is near 1 |

---

## Set 3. Training Challenges and Remedies (Items 21–30)

| No. | Column A: Term |
|---|---|
| 21 | Vanishing gradient |
| 22 | Exploding gradient |
| 23 | Gradient clipping |
| 24 | Truncated BPTT |
| 25 | Skip connections |
| 26 | Dropout |
| 27 | Early stopping |
| 28 | Data leakage |
| 29 | Persistence baseline |
| 30 | Scheduled sampling |

| Letter | Column B: Definition |
|---|---|
| A | Stopping training when the validation score stops improving and keeping the best version |
| B | Backpropagating only k steps within each chunk to save memory |
| C | Stacking several recurrent layers so each layer feeds the next |
| D | Randomly switching off units during training to reduce overfitting |
| E | The backward signal grows without bound, causing loss spikes or NaN |
| F | A naive forecast that predicts tomorrow will equal today |
| G | Rescales an oversized gradient to a maximum norm while keeping its direction |
| H | Information from the future or the test set influences training, for example by shuffling a time series |
| I | Paths that let the signal jump past several steps or layers |
| J | Gradually feeding the model its own predictions during training to reduce exposure bias |
| K | Scaling inputs to zero mean and unit variance before training |
| L | The backward signal shrinks toward zero over many steps, so early steps barely learn |

---

## Set 4. Transfer Learning in CNNs (Items 31–40)

| No. | Column A: Term |
|---|---|
| 31 | Transfer learning |
| 32 | Feature extraction |
| 33 | Fine-tuning |
| 34 | Training from scratch |
| 35 | Negative transfer |
| 36 | Co-adaptation |
| 37 | Gradual unfreezing |
| 38 | Linear probe |
| 39 | ImageNet-1k |
| 40 | Self-supervised pretraining |

| Letter | Column B: Definition |
|---|---|
| A | Training a linear classifier on frozen features to measure their quality |
| B | The exponential of the mean cross-entropy |
| C | Pretraining on unlabeled data with methods such as SimCLR, MoCo and DINO |
| D | A source dataset of about 1.28 million labeled images in 1,000 classes |
| E | Reusing a model trained on a large source task to learn a new target task with less data |
| F | Freezing the pretrained backbone and training only a new head |
| G | Pretrained weights give no gain, or worse results than training from scratch |
| H | Releasing one block per stage, with a lower learning rate for each newly released block |
| I | Continuing to train pretrained layers at a small learning rate |
| J | The gate that controls how much of the cell is exposed as hₜ |
| K | Neighboring layers adapt to each other during source training, so freezing middle layers can hurt |
| L | Starting from random initialization, which needs a large dataset (100k+ images) |

---

## Set 5. Implementation and Practice (Items 41–50)

| No. | Column A: Term |
|---|---|
| 41 | Backbone |
| 42 | Classification head |
| 43 | `requires_grad = False` |
| 44 | `detach()` |
| 45 | `stateful = True` |
| 46 | BatchNorm in inference mode |
| 47 | Channel replication |
| 48 | Patient-level split |
| 49 | Logit gradient (p − y) |
| 50 | Teacher forcing |

| Letter | Column B: Definition |
|---|---|
| A | For softmax with cross-entropy, the predicted probabilities minus the one-hot target |
| B | The PyTorch call that stops gradients from flowing across a chunk boundary in truncated BPTT |
| C | Copying a grayscale image into 3 channels so an RGB-pretrained CNN can accept it |
| D | Memorizing the training set instead of learning general patterns |
| E | The Keras option that carries the hidden state across consecutive batches |
| F | Feeding the ground-truth previous output as the next input during training |
| G | The pretrained convolutional feature extractor, without the final classifier |
| H | Splitting medical images by patient so no patient appears in both train and test |
| I | The final layer(s) that map features to class scores, usually replaced for a new task |
| J | Adding a penalty on large weights to the loss |
| K | Using the stored statistics instead of batch statistics when fine-tuning on small batches |
| L | The PyTorch way to freeze a parameter |

---

# Answer Key

| No. | Ans. | No. | Ans. | No. | Ans. | No. | Ans. | No. | Ans. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | G | 11 | I | 21 | L | 31 | E | 41 | G |
| 2 | D | 12 | A | 22 | E | 32 | F | 42 | I |
| 3 | C | 13 | H | 23 | G | 33 | I | 43 | L |
| 4 | B | 14 | D | 24 | B | 34 | L | 44 | B |
| 5 | I | 15 | E | 25 | I | 35 | G | 45 | E |
| 6 | A | 16 | J | 26 | D | 36 | K | 46 | K |
| 7 | H | 17 | L | 27 | A | 37 | H | 47 | C |
| 8 | L | 18 | B | 28 | H | 38 | A | 48 | H |
| 9 | J | 19 | F | 29 | F | 39 | D | 49 | A |
| 10 | E | 20 | G | 30 | J | 40 | C | 50 | F |

**Extra definitions (match no term)**

| Set | Extras |
|---|---|
| 1 | F (max pooling), K (learning rate) |
| 2 | C (batch normalization), K (same-padding convolution) |
| 3 | C (stacked recurrent layers), K (input standardization) |
| 4 | B (perplexity), J (output gate) |
| 5 | D (overfitting), J (weight decay) |
