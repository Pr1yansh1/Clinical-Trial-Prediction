# Clinical-Trial-Prediction

**Codebase for Clinical Trial Outcome Prediction (TOP)**

Predicting the outcomes of clinical trials is crucial in drug development and healthcare. Using machine learning models, we can provide an estimation of the trial outcomes, optimizing resources and guiding further research.

## Directory Structure

The repository is organized as follows:

- `data/`: Contains all the datasets for training, validation, and testing.
- `data/model.py`: Contains the code for the Random Forest classifier.

### Data

Within the `data/` directory, the datasets are organized per clinical trial phase. For each phase, there are three data files:

- `phase_I_train.csv`
- `phase_I_valid.csv`
- `phase_I_test.csv`
(Repeat for phases II & III)

Each dataset has columns such as `nctid`, `status`, `why_stop`, `label`, `phase`, `diseases`, `icdcodes`, `drugs`, `smiless`, and `criteria`.

### Model

To understand the model's workings and configurations:

- Check `model.py` for the Random Forest classifier's implementation, training, and evaluation procedures.

```python
# To run the model, execute:
python model.py
