🧠 Bigram Language Model (Character-Level)
This project implements a character-level Bigram Language Model using both count-based statistics and a simple neural network to generate realistic-sounding names. The model learns bigram (2-character) probabilities from a corpus of names and can then be used to sample new names character by character.

📂 Files
N-gram Language Model(Bigram).py — Main script implementing data loading, bigram count model, neural bigram model, training loop, and sampling.

names.txt — A text file containing a list of names, one per line (used as training data).

📚 Key Concepts
🔤 Character-Level Modeling
The model operates at the character level (not word level). Each word is treated as a sequence of characters with a start token (.) and an end token (.).

🔁 Bigram Model
A bigram is a pair of consecutive characters (c1, c2). The model learns:

𝑃
(
𝑐
2
∣
𝑐
1
)
P(c 
2
​
 ∣c 
1
​
 )
This is the conditional probability of character c2 following c1.

🔧 Implementation Details
1. Data Preprocessing
Reads names from names.txt

Prepends and appends . to each name to signify start and end.

Builds a vocabulary of 27 characters (a–z and .).

2. Count-Based Bigram Model
Counts all (ch1, ch2) transitions across the dataset.

Stores counts in a 27×27 matrix N.

Converts counts to probabilities using normalization.

3. Visualization
Plots the bigram frequency matrix using matplotlib for visual analysis.

4. Negative Log-Likelihood Evaluation
Calculates the log-likelihood of the dataset under the bigram probability matrix.

Converts it into the average negative log-likelihood (NLL) for evaluation.

5. Neural Bigram Model
Trains a 1-layer neural network:

Inputs: one-hot vectors of characters

Outputs: predicted log-counts for next characters

Optimized using cross-entropy loss and L2 regularization.

6. Sampling
Generates names character-by-character using:

Count-based model

Neural model

Starts from ., samples next character from learned probability distribution until . is generated again.

🧪 Example Output
text
Copy
Edit
maria
jonas
alex.
daren
Generated using the trained neural bigram model.

🛠 Requirements
Python 3.7+

PyTorch

Matplotlib

bash
Copy
Edit
pip install torch matplotlib
🚀 Running the Project
Make sure you have names.txt in the same directory.

Run the Python file:

bash
Copy
Edit
python "N-gram Language Model(Bigram).py"
This will:

Train the model

Plot the frequency matrix

Sample a few names using the trained model

🧩 Potential Extensions
Implement Trigram or N-gram models

Switch to word-level modeling

Integrate LSTM or Transformer for longer dependencies

Evaluate perplexity on a validation set

🤖 Credits
Based on the educational architecture used by Andrej Karpathy in his minimal language modeling tutorials.
