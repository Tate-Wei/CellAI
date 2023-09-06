import logging
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from model.src.optimizer import Optimizer
from model.src.model import Model_Selection
from model.src.dataset import Dataset
from model.src.utils.logger import logger_setup
#from model.src.utils.param import args
from model.src.utils.utils import Usage

RESIZE_SIZE = [256]
CROP_SIZE = [224]
MEAN = [0.48235, 0.45882, 0.40784]
STD = [0.00392156862745098, 0.00392156862745098, 0.00392156862745098]

def main(args_train: bool,
         args_aLearning: bool,
         args_pred: bool,
         args_load: str,
         ):
    # Initialize logger
    logger_setup()
    logger = logging.getLogger(__name__)

    # Train and validation datasets
    assert not (args_train and args_aLearning and args_pred), "Perform either training or active learning or prediction."
    assert not (not args_train and not args_aLearning and not args_pred), "Perform either training or active learning or prediction."

    transform_imagenet = transforms.Compose([
        transforms.Resize(RESIZE_SIZE, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    if args_train:
        training_set = Dataset(Usage.TRAINING, transform_imagenet)
    elif args_aLearning:
        training_set = Dataset(Usage.A_LEARNING, transform_imagenet)
    elif args_pred:
        pred_set = Dataset(Usage.PREDICTION, transform_imagenet)

    # Create model
    model = Model_Selection(args_load)

    if args_train or args_aLearning:
        # Start training
        optimizer = Optimizer(Usage.TRAINING, training_set, model.model, 0.8)
        optimizer.train()

    elif args_pred:
        optimizer = Optimizer(Usage.PREDICTION, pred_set, model.model, None)
        optimizer.prediction()

