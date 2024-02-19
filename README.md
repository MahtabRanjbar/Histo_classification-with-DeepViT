
# Histopathology Image Classification with DeepVit on BreakHis Dataset

This project leverages the DeepVit model, a variant of the Vision Transformer (ViT), for the classification of histopathology images from the BreakHis dataset. The BreakHis dataset consists of microscopic biopsy images of benign and malignant breast tumors, and the DeepVit model is used to automate the differentiation between these tumor types.  

<br>


## Project Structure

- **`data_preparation.py`**: Downloads, extracts, and organizes the BreakHis dataset.
- **`model.py`**: Defines the DeepVit model architecture tailored for histopathology image classification.
- **`train_and_evaluate.py`**: Manages the training process, evaluates model performance, and plots training/validation metrics.
- **`config.py`**: Contains model parameters, training settings, and other configurations.
- **`main.py`**: The main script that orchestrates the data preparation, model training, and evaluation processes.

<br>

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Training the Model**

    ```bash
    python src/main.py
    ```

## Dataset Overview

The BreakHis dataset comprises 7,909 microscopic images of breast tumor tissue, collected from 82 patients. Images are labeled as either benign or malignant, with further subclassifications within each category. The dataset images are at different magnifications (40X, 100X, 200X, and 400X).

Before training the model, run the `data_preparation.py` script to download and organize the dataset properly:

```bash
python data_preparation.py
```

## Model Architecture

DeepVit enhances traditional Vision Transformers by addressing the "attention collapse" issue found in deeper architectures. It introduces a "Re-attention" mechanism, significantly diversifying attention maps with minimal extra cost. This advancement allows DeepVit to effectively scale and improve performance in image classification, making it particularly suited for the nuanced analysis required by the BreakHis histopathology dataset.



This script will automatically handle dataset preparation, model initialization, training, and evaluation.

## Results

Upon training completion, the project generates a detailed report, which includes:

- Training and validation loss and accuracy across epochs.
- The total number of trainable parameters in the DeepVit model. 


| Metric    | Value     |
|-----------|-----------|
| Accuracy  | 83.55 |
| Recall    | 0.83  |
| Precision | 0.83  |
| F1 score  | 0.82  |


Find the results in  [Report](./report) folder.

## Contributing

Contributions to this project are welcome. Feel free to fork the repository, make enhancements, and submit pull requests.

## License

This project is released under the MIT License. See the LICENSE file for more details.



