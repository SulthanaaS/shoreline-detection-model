import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from data_loading import get_dataset  # Assuming you have a function to load datasets
from deep_learning import get_model, Metrics, full_forward, flatui_cmap  # Assuming you have necessary functions and classes

@torch.no_grad()
def test(dataset, model, config):
    # Test step
    data_loader = DataLoader(dataset,
        batch_size=config['batch_size'],
        shuffle=False, num_workers=config['data_threads'],
        pin_memory=True
    )

    model.train(False)
    idx = 0
    for img, target in tqdm(data_loader):
        B = img.shape[0]
        res = full_forward(model, img, target, Metrics())  # Create a new Metrics object for each batch

        for i in range(B):
            # Perform any additional evaluation or visualization if needed
            pass

        idx += B

    # Calculate and print evaluation metrics if needed
    metrics_vals = res['metrics'].evaluate()
    logstr = f'Test: ' \
           + ', '.join(f'{key}: {val:.3f}' for key, val in metrics_vals.items())
    print(logstr)

# Load the configuration file
config_file = Path('config.yml')  # Update the path if needed
config = yaml.load(config_file.open(), Loader=yaml.SafeLoader)

# Load the model
modelclass = get_model(config['model'])
model = modelclass(**config['model_args'])

# Load the trained model checkpoint
checkpoint_path = Path(r'C:\Users\USER\Downloads\HED-UNet-master_timex\HED-UNet-master\logs\2024-05-02_12-25-57\checkpoints\10.pt')  # Update the path to your trained checkpoint
model.load_state_dict(torch.load(checkpoint_path))

# Load the test dataset
test_dataset = get_dataset('test')  # Assuming you have a function to load test dataset

# Call the test function
test(test_dataset, model, config)
