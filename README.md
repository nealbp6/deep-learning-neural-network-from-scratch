# Deep Learning Practice (FFNN)

## Overview

This repository contains a from scratch created feedforward neural network (FFNN) to demonstrate and learn core deep learning concepts. Built entirely by hand with NumPy, this project showcases fundamental neural network architecture, training mechanics, and evaluation practices—ideal for learning and portfolio demonstration.

**Note:** A detailed technical documentation PDF accompanies this project with in-depth explanations of the architecture, math, and implementation details.

---

**Quick start**

download all python files

Install dependencies:

```bash
pip install numpy matplotlib yfinance
```

Run the main script:

```bash
python main.py
```

The script downloads price data with `yfinance`, computes log returns, normalizes and windows the data, trains the FFNN, evaluates metrics, and plots predictions.

---

## How to Customize Parameters

Edit the following variables in `main.py` to experiment and customize the model behavior:

- **`symbol`, `interval`**: Change the  ticker and data frequency fetched via `yfinance`.
- **`window_size`**: Adjust the length of the input window used for each prediction.
- **`train_ratio`**: Modify the train/test split ratio.
- **`learning_rate`**: Control the SGD step size during training.
- **`epochs`**: Set the number of training iterations.
- **Hidden layer sizes** (in `ffnn.py`): Adjust `n_hidden1`, `n_hidden2`, `n_hidden3` to experiment with network depth and width, you can also add new layers.


## AI Disclosure

During the development of this Deep Learning project, AI-powered tools were used to assist in structuring, debugging, and documenting code. Specifically:

- **ChatGPT and other models:** Assisted in generating explanations, improving readability, and optimizing code flow.

> ⚠️ **Important:** All logic, architecture, and implementation were designed, reviewed, and tested by the human developer. AI tools were used only to assist productivity — not to generate autonomous or unverified code.

## License

This project is licensed under the MIT License.

> © **2025 Neal**  

> *Feel free to modify, improve, and share — just include credit to the original author.*

