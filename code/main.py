import os
import torch
import pandas as pd
import numpy as np
from types import SimpleNamespace
from torch.utils.data import DataLoader

from src.loss_fn import RMSELoss
from src.dataloading import load_data, Dataset
from src.model import MLP
from src.trainer import Trainer
from src.utils import set_seed, write_dict_to_excel, parse_list
from src.callbacks import ModelCheckpoint

set_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = '../data'
results_dir = '../results'
os.makedirs(f'{results_dir}/', exist_ok=True)

# Load dataset
x_train, y_train, x_val, y_val, x_test, y_test = load_data(dir_data=data_dir)
train_dataset = Dataset(x_train, y_train)
val_dataset = Dataset(x_val, y_val)
test_dataset = Dataset(x_test, y_test)

def train_model(model_name, hyperparams, model_params, callbacks=[None, None]):
    set_seed(42)
    
    train_dl = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    model = MLP(model_params[model_name], input_shape=hyperparams['input_shape']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])
    criterion = RMSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparams['epochs'], eta_min=1e-10)

    trainer = Trainer(hyperparams['epochs'], criterion, optimizer, scheduler, callbacks)
    history = trainer.train(train_dl, val_dl, model, device)
    return history, model, trainer

def main(model_name, fold, hyperparams, model_params):
    os.makedirs(f'{results_dir}/{model_name}', exist_ok=True)
    history, _, _ = train_model(model_name, hyperparams, model_params)

    idx = np.argmin(np.array(history['val_loss']) + np.array(history['train_loss']))
    loss = history['val_loss'][idx] + history['train_loss'][idx]
    xlsx_record = {
                    'metric': loss,
                    'train_loss': history['train_loss'][idx],
                    'val_loss': history['val_loss'][idx],
                    'learning_rate': hyperparams['learning_rate'],
                    'weight_decay': hyperparams['weight_decay'],
                    }
    xlsx_record = xlsx_record | model_params[model_name]
    write_dict_to_excel(data=pd.DataFrame([xlsx_record]), filename=f'{results_dir}/{model_name}/{fold}.xlsx', sheet_name='experiment')
    return loss

def test(model_name, fold, hyperparams, model_params):
    df = pd.read_excel(f'{results_dir}/{model_name}/{fold}.xlsx', sheet_name='experiment')
    opt_params = df.loc[df['metric']==df['metric'].min()]

    # Assign values to hyperparameters and model parameters
    hyperparams['learning_rate'] = float(opt_params['learning_rate'].iloc[0])  # Convert single values explicitly
    hyperparams['weight_decay'] = float(opt_params['weight_decay'].iloc[0])
    model_params['mlp']['units'] = parse_list(opt_params['units'].iloc[0])  # Already a list
    model_params['mlp']['drop_coef'] = float(opt_params['drop_coef'].iloc[0])
    model_params['mlp']['activation'] = opt_params['activation'].iloc[0]

    callbacks = [ModelCheckpoint(filepath=f'{results_dir}/{model_name}/Model_{fold}.pt', epoch_start=hyperparams['epoch_start'], verbose=True), None]
    _, model, train_obj = train_model(model_name, hyperparams, model_params, callbacks)

    test_dl = DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    model.load_state_dict(torch.load(f'{results_dir}/{model_name}/Model_{fold}.pt', weights_only=True))
    model.eval()

    df = train_obj.test(test_dl, model, device)
    filename = f'{results_dir}/{model_name}_test.xlsx'
    write_dict_to_excel(data=df, filename=filename, sheet_name='experiment')
    
if __name__=='__main__':
    # ==========================================
    # EXPERIMENT CONFIGURATION
    # Modify these parameters to adjust training
    # ==========================================
    model_name = 'mlp'
    fold = '0'

    hyperparams = {
        'learning_rate':1e-3,
        'batch_size':32,
        'epochs':100,
        'input_shape':1001,  
        'epoch_start':0,
        'weight_decay':1e-4,
        'patience':5,
        'optimizer':'AdamW',
        'criterion':'RMSELoss' 
    }

    model_params = {
        'mlp': {
            'units': [128, 64],
            'n_outputs': 7, 
            'activation': 'relu',
            'drop_coef': 0.5,
            'classify': False
        }
    }

    print(f"Starting experiment with fold: {fold}")
    main(model_name, fold, hyperparams, model_params)
    print("Optimization completed. Running final test inference...")
    test(model_name, fold, hyperparams, model_params)
