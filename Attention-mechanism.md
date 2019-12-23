
# Attention mechanism

## The papers

Neural Machine Translation by Jointly Learning to Align and Translate https://arxiv.org/abs/1409.0473 (the attention paper)

Attention? Attention! https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html

Effective Approaches to Attention-based Neural Machine Translation https://arxiv.org/pdf/1508.04025.pdf (global attention vs local attention; ie, looking at all words vs looking at some words)

Show, Attend and Tell: Neural Image CaptionGeneration with Visual Attention http://proceedings.mlr.press/v37/xuc15.pdf (attention for image)

### ELI5

When you first look at an image, you first focus on one spot before moving to another spot. This wandering focus is your attention: it spotlights the things that matter, and ignores the things that don't.

Just as you can divide an image into sections (spots), and selectively process on just a few of them, you can do the same thing with words, where each word token is its own section.

Besides scanning from spot to spot on an image, your mind also 'expects' (predicts) what it'll see next based on the things it has already seen. For example, when you see a furry texture with a snout, your mind expects to see an animal.

With words, if you read 'she is eating a ...', you'd expect that whatever follows is a food of some sort.

So attention is not just the focus on where to look, but also how the item you're looking at change your expectation because things in a scene can relate to each other.

You can do the same to a RNN encoder-decoder model for words:

Retain all the hidden states, and stack them into a matrix

Apply a weight to select which hidden state to look at when the decoder is generating a translation

### The encoder

The input sentence is run twice: once forwrad, once backward into the encoder. The hidden states of each run are collected and lined up according to the order of the sentence.

Now, each word in the sentence has 2 hidden states: one that remembers the words before it, and the other remembers the words that come after it.

The run from the first word $x_1$ to the last word $x_T$ to calculate the forward hidden states:

$$(\overrightarrow{h_1}, \dots, \overrightarrow{h_{T_x}})$$

The backward run from the last word $x_T$ to the first word $x_1$ to calculate the backward hidden states:

$$(\overrightarrow{h_{T_x}}, \dots, \overrightarrow{h_1})$$

This backward hidden states vector is reversed so that the order of inputs $x_t$ are aligned with the forward hidden states vector:

$$(\overleftarrow{h_1}, \dots, \overleftarrow{h_{T_x}})$$

The bidirectional hidden states for each input word are stacked (concatenated) together to form __annotation__ $h_j$:

$$h_j = \left[
\overrightarrow{h}_j^\top ; \overleftarrow{h}_j^\top \right]^\top$$

Since RNNs only retain recent knowledge, the bidirectional hidden states at each step will have information for both the preceding words, and the following words.

### The decoder

The decoder decides parts of the source sentence to pay attention to.

#### The attention

To find which words (annotations) to pay attention to, we need some sort of mask to highlight certain words, and diminish others before the decoder can decide on an output.

This mask is the context vector $c_i$ (aka, the attention)

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$$

The context vector __for the current step__ $c_i$ is the sum of the current step's alignment weight $a_{ij}$ multiplied by __all annotations__ $h_j$.

_Alignment_ means matching the input word (annotation) to the output word. The alignment is based on the decoder's previous hidden state $s_{i-1}$ and the annotations $h_j$.

__$a_{ij}$ is the weights that select which words to pay attention to by highlighting some words, and diminishing others__.

$a_{ij}$ is the probability that output $y_i$ is aligned to the source word $x_j$.

For each output $y_i$, an annotation weight $a_{ij}$ is generated for every word $x_{j}$ of the source sentence.

Once the decoder has the context vector, it can go on to calculate the new hidden state $s_i$ and an output $y_i$:

$$s_i = f(s_{i-1}, y_{i-1}, c_i)$$

$$y_i = g(s_i, y_{i-1}, c_i)$$

~~Before the decoder generates an output $y_t$, it needs to figure out which input word to look at. So the decoder calculates a context vector $c_i$ to decide which words to look at. Once it has a context vector, the decoder can take the previous input, and the previous step's hidden state, and the context vector to produce a new output $y_t$.~~

#### TODO

I still don't fully understand how the annotation weight $a_{ij}$ is calculated. The paper lists these 2 equations which I still can't parse in my head (in plain language), so I'll leave them here and update my notes later when I finally figure them out.

Copied verbatim from the paper:

$$\alpha_{ij} = \frac{\exp\left(e_{ij}\right)}{\sum_{k=1}^{T_x} \exp\left(e_{ik}\right)}$$

where

$$e_{ij} = a(s_{i-1}, h_j)$$

is an _alignment model_ which scores how well the inputs around position $j$ and the output at position $i$ match. The score is based on the decoder's last hidden state $s_{i-1}$

---

From [Show, Attend and Tell: Neural Image CaptionGeneration with Visual Attention](http://proceedings.mlr.press/v37/xuc15.pdf) by Xu et, al:

For each location $i$, the attention mechanism generates a positive weight $\alpha_i$ which can be interpreted either as the probability that location $i$ is the right place to focus for producing the next word, or as the relative importance to give to each location.

The weight $\alpha_i$ of each annotation vector $a_i$ is computed by an attention model $f_{att}$ for which we use a multilayer perceptron conditioned on the previous hidden state $h_{t−1}$. To emphasize, we note that the hidden state varies as the output RNN advances in its output sequence: "where" the network looks next depends on the sequence of words that has already been generated.

# Other forms of attention

self-attention: association/correlation between the current word and the previous parts of the sentence

soft vs hard attention: soft attention looks at all words of a sentence while hard attention selects only 1.

global vs local: global is like soft attention while local is attention localized (like a window) around the source word and is used to calculate the context vector.

transformer:

1. self attention: how this word relates to others;
1. feed forward network: the weights

---

### how to calculate self-attention

1. convert input embeddings into 3 new vectors: queries $Q$, keys $K$, and values $V$ by multiplying by their corresponding weight matricies $W_Q$, $W_K$, $W_V$ whose values are learned during training
1. calculate a score of how every input relates to the current one, including itself. the scores are the dot product between the current word's query $q_1$ and every word's key $k_j$. This includes the current key $k_1$. So if you have 3 input words, and you're processing the first word, you'd have 3 scalar scores at this stage: $q_1 \cdot k_1$; and $q_1 \cdot k_2$; and $q_1 \cdot k_3$
1. divide each scalar score by the square root of the dimension of the key vectors. For example, if the dimension of the key vectors is 64, then divide the scalars by $\sqrt{64}$, or 8. This leads to more stable gradients.
1. normalize the scalars through a softmax so that they're between 0 and 1, and that they all add up to 1. Obviously, the current word will have the highest score.
1. Multiply all the values $v_j$ by the softmax score. This will diminish or amplify certain inputs.
1. Sum all the vectors from the previous step. This is the self-attention $z_1$ of the current word.

The above process are repeated 8 separate times concurrently, with 8 separate sets of weights and results. Each run is a 'head', in multi-headed attention. Think of an 8-headed frankenstein monster, each head reads the input sentence and has its own interpretation.

These 8 outputs of $z$ are stacked (concatenated), and multitplied by an additional weight matrix $W^O$ (trained with the model). The result is a $Z$ matrix that captures the attention of all 8 heads.

