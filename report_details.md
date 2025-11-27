# Theoretical Framework & Implementation Details

This document provides the theoretical background and mathematical formulations for the components used in the Unified Stock Prediction Pipeline.

## 1. Sentiment Analysis (TextBlob)

We utilize **TextBlob**, a lexicon-based sentiment analysis tool, to quantify the qualitative information present in news articles.

### Theoretical Basis
TextBlob calculates sentiment based on a pre-defined lexicon where words are annotated with semantic scores. It computes two metrics:

1.  **Polarity ($P$)**: Measures the sentiment orientation (Positive vs. Negative).
    *   **Range**: $[-1.0, +1.0]$
    *   $-1.0$: Extremely Negative
    *   $0.0$: Neutral
    *   $+1.0$: Extremely Positive
2.  **Subjectivity ($S$)**: Measures the amount of personal opinion vs. factual information.
    *   **Range**: $[0.0, 1.0]$
    *   $0.0$: Objective (Fact)
    *   $1.0$: Subjective (Opinion)

### Mathematical Formulation
For a given text document $D$ containing words $w_1, w_2, ..., w_n$:

$$ P(D) = \frac{\sum_{i=1}^{n} \text{valence}(w_i) \cdot \text{modifier}(w_{i-1})}{\sum_{i=1}^{n} \mathbb{I}(w_i \in \text{Lexicon})}$$

Where:
*   $\text{valence}(w_i)$ is the pre-computed polarity score of word $w_i$ from the lexicon.
*   $\text{modifier}(w_{i-1})$ accounts for intensifiers (e.g., "very", "slightly") or negations (e.g., "not").
    *   If "not" precedes $w_i$, the polarity is flipped: $P' = P \times -0.5$ (approximate heuristic).
    *   If "very" precedes $w_i$, the intensity is increased.

In our pipeline, we aggregate these scores daily:
$$ P_{daily} = \frac{1}{N} \sum_{j=1}^{N} P(Article_j) $$

## 2. Reinforcement Learning (RL) Component

We implement a **Directional Policy Gradient-inspired Loss** to align the model's objectives with trading profitability (buying low, selling high).

### Concept
Standard regression models minimize Mean Squared Error (MSE), which focuses on the *magnitude* of the error. However, in trading, the *direction* of the movement is often more critical than the exact price.
*   **MSE Goal**: Minimize $(y_{true} - y_{pred})^2$
*   **RL Goal**: Maximize Profit $\approx$ Correct Direction Prediction.

### Mathematical Formulation
We define a composite loss function $L_{total}$ that combines regression accuracy with a directional penalty.

$$ L_{total} = L_{MSE} + \lambda \cdot L_{Directional} $$

#### 1. MSE Term ($L_{MSE}$)
Standard Euclidean distance minimization:
$$ L_{MSE} = \frac{1}{N} \sum_{t=1}^{N} (y_t - \hat{y}_t)^2 $$

#### 2. Directional Penalty ($L_{Directional}$)
We want to penalize the model when the predicted change $\Delta \hat{y}_t$ has a different sign than the actual change $\Delta y_t$.

Let:
$$ \Delta y_t = y_t - y_{t-1} \quad (\text{Actual Move}) $$
$$ \Delta \hat{y}_t = \hat{y}_t - \hat{y}_{t-1} \quad (\text{Predicted Move}) $$

We define a "match" function using the sign operation. To make it differentiable for backpropagation (required for Neural Networks), we approximate the sign function using the hyperbolic tangent ($\tanh$):

$$ \text{sign}_{soft}(x) \approx \tanh(k \cdot x) $$
*Where $k$ is a scaling factor (e.g., 10) to make the transition steep.*

The directional alignment score $A_t$ is the product of the signs:
$$ A_t = \tanh(k \cdot \Delta y_t) \cdot \tanh(k \cdot \Delta \hat{y}_t) $$
*   If both go UP (+ * +) $\rightarrow A_t \approx 1$
*   If both go DOWN (- * -) $\rightarrow A_t \approx 1$
*   If mismatch (+ * -) $\rightarrow A_t \approx -1$

We want to **minimize** the penalty, so we define the loss as:
$$ L_{Directional} = \frac{1}{N} \sum_{t=1}^{N} (1 - A_t) $$
*   If correct ($A_t=1$), Loss = 0.
*   If wrong ($A_t=-1$), Loss = 2.

### Final Loss Function
$$ L = \frac{1}{N} \sum (y - \hat{y})^2 + \lambda \left( 1 - \tanh(10 \Delta y) \cdot \tanh(10 \Delta \hat{y}) \right) $$

This forces the model to learn the trend (RL reward) while staying close to the true price (MSE constraint).

## 3. xLSTM-Inspired Architecture

The model architecture is designed to capture long-term dependencies better than standard LSTMs, inspired by the principles of Extended LSTM (xLSTM).

### Key Features
1.  **Residual Connections**:
    $$ h_{out} = h_{in} + \text{LSTM}(h_{in}) $$
    This allows gradients to flow through the network more easily, mitigating the vanishing gradient problem and allowing deeper stacking of layers.

2.  **Layer Normalization**:
    Applied after each LSTM block. It stabilizes the hidden state dynamics:
    $$ \hat{h} = \frac{h - \mu}{\sigma} \cdot \gamma + \beta $$
    This ensures that the inputs to the next layer (and the RL loss gradients) remain well-scaled, which is crucial when combining distinct modalities like Price (continuous) and Sentiment (bounded).

### Multi-Modal Fusion
The inputs are concatenated before entering the network:
$$ X_t = [ \text{Price}_t, \text{Volume}_t, \text{Sentiment}_t, \text{Subjectivity}_t ] $$
This allows the LSTM gates (Input, Forget, Output) to condition their state updates on both market technicals and news sentiment simultaneously.

## 4. Portfolio Optimization Strategy (Daily Selection)

To maximize daily profit, we extend the pipeline from single-stock prediction to **Cross-Sectional Portfolio Ranking**.

### Objective
Identify the single most profitable asset to **Buy** (Long) and the single most likely asset to drop to **Sell** (Short) for each trading day $t$.

### Mathematical Formulation
Let $\mathcal{S}$ be the set of available tickers (e.g., INFY, TCS, RELIANCE).
For each stock $i \in \mathcal{S}$, the model predicts the closing price $\hat{P}_{i,t}$.

We calculate the **Expected Return** $\hat{r}_{i,t}$:
$$ \hat{r}_{i,t} = \frac{\hat{P}_{i,t} - P_{i, t-1}}{P_{i, t-1}} $$

### Selection Policy (Greedy Max-Min)
The RL agent's policy $\pi(s_t)$ selects actions $a_t = (\text{Buy}_i, \text{Sell}_j)$ to maximize the daily reward $R_t$.

1.  **Buy Signal (Long)**: Select the stock with the highest positive expected return.
    $$ i^*_{buy} = \operatorname*{argmax}_{i \in \mathcal{S}} (\hat{r}_{i,t}) $$
    *Constraint*: $\hat{r}_{i^*_{buy}, t} > \tau$ (Threshold, e.g., 0)

2.  **Sell Signal (Short)**: Select the stock with the lowest (most negative) expected return.
    $$ j^*_{sell} = \operatorname*{argmin}_{j \in \mathcal{S}} (\hat{r}_{j,t}) $$
    *Constraint*: $\hat{r}_{j^*_{sell}, t} < -\tau$

### Portfolio Reward Function
The refined RL objective is to maximize the **Realized Portfolio Return**:
$$ J(\theta) = \sum_{t} \left( r_{i^*_{buy}, t} - r_{j^*_{sell}, t} \right) $$
Where $r_{i,t}$ is the *actual* return of stock $i$ at time $t$. This transforms the problem from minimizing individual regression errors to maximizing the spread between the best and worst performers.

