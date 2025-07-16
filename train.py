import torch
from fastai.vision.all import *
from fastai.data.transforms import *
from torchvision.datasets import MNIST

def create_mnist_model():
    # Download MNIST dataset
    path = Path('Data')
    path.mkdir(exist_ok=True)
    
    # Download MNIST if not exists
    if not (path/'training').exists():
        MNIST(root=path, download=True)
        # Move downloaded files to expected structure
        (path/'training').mkdir(exist_ok=True)
        (path/'testing').mkdir(exist_ok=True)
        
        # Move train and test images
        for file in (path/'MNIST'/'raw').glob('*train*'):
            shutil.move(str(file), str(path/'training'))
        for file in (path/'MNIST'/'raw').glob('*t10k*'):
            shutil.move(str(file), str(path/'testing'))
    
    # Create DataLoaders
    dls = ImageDataLoaders.from_folder(
        path, 
        train='training', 
        valid='testing', 
        item_tfms=Resize(28),
        batch_tfms=aug_transforms(size=28)
    )
    
    # Create model
    learn = cnn_learner(dls, resnet18, metrics=accuracy)
    
    # Train the model
    learn.fit_one_cycle(5)
    
    # Save the model
    learn.export('mnist_model.pkl')
    
    return learn

if __name__ == '__main__':
    model = create_mnist_model()
    print("Model training completed and saved.")