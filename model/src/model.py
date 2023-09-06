import torch, logging, os
import torchvision.models as models

from model.src.utils.utils import load_settings

logger = logging.getLogger(__name__)

class Model_Selection():
    def __init__(self, args_load: str) -> None:

        self.checkpoint_path = args_load
        self.model = models.vgg16(weights='IMAGENET1K_V1')

        self._freeze_param()
        if self.checkpoint_path:
            mode = load_settings()['mode']
            path = os.path.join(self.checkpoint_path, f'checkpoint_{mode}.pth')
            self.model.load_state_dict(torch.load(path))

    def _freeze_param(self) -> None:
        """Freeze the model parameters"""

        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = torch.nn.Linear(num_features, 5) 
        for param in self.model.features.parameters():
            param.requires_grad = False
