# BERT-LSTM-CNN
## Data
The csv datasets could be downloaded here: [REDD](http://redd.csail.mit.edu/) and [UK-DALE](https://jack-kelly.com/data/)
## Training
Run following command to train an initial model, test will run after training ends:
```bash
python train.py
```
The trained model state dict will be saved under 'experiments/dataset-name/best_acc_model.pth'
## Evaluation metrics
The code of evaluation matrics in our paper was illustrated here, including ACC, F1, MRE, MAE.
