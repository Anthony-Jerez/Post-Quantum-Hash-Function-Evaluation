# Post-Quantum Hash ML Attack Demo

This repository provides a demonstration of a machine learning–assisted pre-image attack on a post-quantum hash function on the parameter set: ($n=3$, $a=4$, $b=2$, walk length $\ell=4$ and prime $p=5$). It's based on the research paper *“Post‑quantum hash functions using $\mathrm{SL}_n(\mathbb F_p)$”* written by Corentin Le Coz, Christopher Battarbee, Ramon Flores, Thomas Koberda, and Delaram Kahrobaei. It includes scripts to:

1. **Construct the hash function** in Python and verify it against the published example.  
2. **Generate a full dataset** of all 3^8 input sequences and their corresponding hash outputs.  
3. **Preprocess** the data into PyTorch `TensorDataset` format.  
4. **Train** a lightweight PyTorch LSTM model to approximate the hash function.  
5. **Evaluate** the model’s performance via standard MSE and a search-space reduction metric simulating an attacker.  

---

## Repository Structure

```
├── data/
│   ├── all_inputs.npy       # Raw input sequences
│   ├── all_outputs.npy      # Raw hash outputs
│   ├── dataset.pth          # PyTorch TensorDataset
│   └── dataloader.pth       # Prebuilt DataLoader (optional)
├── notebooks/
│   └── post_quantum_hash_demo.ipynb   # Jupyter notebook walkthrough
├── scripts/
│   ├── make_hash.py         # Hash construction example
│   ├── gen_dataset.py       # Enumerate sequences, save .npy
│   ├── preprocess.py        # Convert .npy to TensorDataset
│   └── train_attack.py      # LSTM training & evaluation script
└── README.md
```

---

## Requirements

- Python 3.8+  
- NumPy  
- PyTorch 2.x   

Install dependencies via:

```bash
pip install numpy torch
```

---

## Usage

1. **Hash Construction**  
   ```bash
   python scripts/make_hash.py
   ```  
   Verifies the toy hash matches the paper's example.

2. **Dataset Generation**  
   ```bash
   python scripts/gen_dataset.py
   ```  
   Creates `all_inputs.npy` and `all_outputs.npy` for 3^8 sequences.

3. **Preprocessing**  
   ```bash
   python scripts/preprocess.py
   ```  
   Saves `dataset.pth` containing a PyTorch `TensorDataset`.

4. **Training & Evaluation**  
   ```bash
   python scripts/train_attack.py
   ```  
   - Splits data (train/probe)  
   - Trains an LSTM on train set  
   - Reports train MSE  
   - Evaluates probe MSE and average search-space reduction  

5. **Interactive Notebook**  
   Launch Jupyter:  
   ```bash
   jupyter notebook notebooks/post_quantum_hash_demo.ipynb
   ```  
   Walk through code and metrics.

---

## Results

With a **50% / 50% train–probe split**, we observed the following:

- **Train MSE:** ~0.74  
- **Probe MSE:** ~0.90  
- **Average search-space reduction:** ~93.9%

These results confirm the model's ability to generalize effectively from half the data and significantly reduce brute-force effort on a toy post-quantum hash instance.
