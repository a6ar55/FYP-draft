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

## 4. Portfolio Optimization Strategy (Long-Term Trend Following)

To mitigate the noise inherent in daily price movements, we shift from a daily-trading approach to a **Long-Term Trend Following** strategy.

### Objective
Identify assets with the strongest **30-Day Forward Return Potential**. Instead of frequent churning, we allocate capital to the asset with the most robust medium-term growth trajectory.

### Mathematical Formulation
We modify the prediction target. Instead of predicting $P_{t+1}$, the model now predicts the price at horizon $H$ (e.g., $H=30$ days).

$$ \hat{P}_{i, t+H} = f_\theta(X_t) $$

The **Expected Horizon Return** is:
$$ \hat{R}_{i, t \to t+H} = \frac{\hat{P}_{i, t+H} - P_{i, t}}{P_{i, t}} $$

### Selection Policy (Rolling Best-Idea)
At each time step $t$, we rank all available assets based on their forecasted 30-day return.

1.  **Ranking**:
    $$ \text{Rank}_t = \operatorname*{argsort}_{i \in \mathcal{S}} (\hat{R}_{i, t \to t+H}) $$

2.  **Allocation**:
    We invest 100% of the portfolio into the Top-1 ranked asset ($i^*$).
    $$ i^*_t = \operatorname*{argmax}_{i \in \mathcal{S}} (\hat{R}_{i, t \to t+H}) $$

3.  **Holding Logic**:
    This naturally implements a "hold" strategy. Since 30-day trends evolve slowly, $i^*_t$ is likely to remain $i^*_{t+1}$ for many consecutive days. We only switch assets (sell old, buy new) when a new asset overtakes the current holding in terms of predicted medium-term potential.

### Reward Function
The simulation tracks the **Daily Realized Return** of the selected asset:
$$ \text{Profit}_t = \frac{P_{i^*_t, t+1} - P_{i^*_t, t}}{P_{i^*_t, t}} $$
*Note: Even though we predict 30 days out, we re-evaluate daily to ensure we are always in the best asset.*

## 5. Risk Management (ATR Trailing Stop)

To protect capital from significant drawdowns, we implement a dynamic **Volatility-Based Stop Loss** (Chandelier Exit).

### Concept
Fixed percentage stop-losses (e.g., 5%) are often inefficient because they don't account for the asset's natural volatility. A 5% drop in a stable stock is a crash, but in a volatile stock, it's noise.
We use the **Average True Range (ATR)** to adapt the stop-loss distance to the current market volatility.

### Mathematical Formulation
We maintain a **Trailing Stop Price** ($P_{stop}$) that only moves *up*, never down.

$$ P_{stop, t} = \max(P_{stop, t-1}, P_{close, t} - k \cdot \text{ATR}_t) $$

Where:
*   $k$: Multiplier (typically 2.0 to 3.0). We use **2.0**.
*   $\text{ATR}_t$: The 14-day Average True Range at time $t$.

### Execution Logic
At each time step $t$:
1.  **Check Stop Condition**: If $P_{low, t} < P_{stop, t-1}$:
    *   **SELL IMMEDIATELY** (Stop Loss Triggered).
    *   Move to **Cash** for the remainder of the day.
    *   Reset holding.
2.  **Update Stop Price**: If still holding, update $P_{stop, t}$ using the formula above.
3.  **Model Re-evaluation**: If not stopped out, check if the model predicts a better asset. If so, switch (Sell current, Buy new).

This ensures we let profits run (by trailing the price up) but cut losses quickly when the trend breaks by more than $2 \times$ Volatility.

