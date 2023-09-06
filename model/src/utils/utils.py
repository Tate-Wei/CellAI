import os, logging
from dataclasses import dataclass
import numpy as np

#from src.utils.param import args
from ami.settings import BASE_DIR

ST_PATH = BASE_DIR / 'model' / 'settings' / 'settings.npy'
logger = logging.getLogger(__name__)

@dataclass
class Usage():
    TRAINING: str='training'
    VALIDATION: str='validation'
    A_LEARNING: str='a_learning'
    PREDICTION: str='prediction'

class Document():

    @staticmethod
    def docu_eval_hist(content: dict,
                       file_name: str,
                       mode: str='w',
                       shown=False) -> None:
        """Document the evaluation result"""

        if not os.path.exists(BASE_DIR / 'model' / 'outputs' / "logs"):
            os.makedirs(BASE_DIR / 'model' / 'outputs' / "logs")
        
        try:
            epoch_idx = content['epoch']
            file_path = BASE_DIR / 'model' / 'outputs' / "logs" / f"{file_name}_epoch_{epoch_idx}.log"
        except KeyError:
            file_path = BASE_DIR / 'model' / 'outputs' / "logs" / f"{file_name}_epoch_X.log"
                
        with open(file_path, mode) as f:
            for key in content.keys():
                f.write(f"{key.replace('_', ' ').title()}: \t{str(content[key])}")
                f.write("\n")
            f.write("\n")
            f.flush()

        if shown:
            text = ""
            for key in content.keys():
                text += f"{key.replace('_', ' ').title()}: \t{str(content[key])}"
                text += "\n"
            text += "\n"

            logger.info(text)

    @staticmethod
    def docu_eval_pred(content: dict,
                       file_name: str='prediction',
                       mode: str='w',
                       shown=False) -> None:
        """Document the prediction result"""

        if not os.path.exists(BASE_DIR / 'model' / 'outputs' / "logs"):
            os.makedirs(BASE_DIR / 'model' / 'outputs' / "logs")
        
        file_path = BASE_DIR / 'model' / 'outputs' / "logs" / f"{file_name}_result.log"
                
        with open(file_path, mode) as f:
            for key in content.keys():
                f.write(f"{key.replace('_', ' ').title()}: \t{str(content[key])}")
                f.write("\n")
            f.write("\n")
            f.flush()

        if shown:
            text = ""
            for key in content.keys():
                text += f"{key.replace('_', ' ').title()}: \t{str(content[key])}"
                text += "\n"
            text += "\n"

            logger.info(text)

    @staticmethod
    def docu_training_loss_hist(content: dict, 
                                file_name: str='training_loss_hist', 
                                mode: str='w', 
                                shown=False) -> None:
        """Document the loss history"""

        if not os.path.exists(BASE_DIR / 'model' / 'outputs' / "logs"):
            os.makedirs(BASE_DIR / 'model' / 'outputs' / "logs")
        
        file_path = BASE_DIR / 'model' / 'outputs' / "logs" / f"{file_name}_epoch_{content['epoch']}.log"
        with open(file_path, mode) as f:
            f.write(f"Loss in epoch: \t{str(content['epoch'])}")
            f.write("\n")
            f.write(f"Average loss: \t{str(content['avg_tr_loss'])}")
            f.write("\n")
            f.flush()

        if shown:
            text = f"Loss in epoch: \t{str(content['epoch'])}"
            text += f"\n"
            text += f"Average loss: \t{str(content['avg_tr_loss'])}"
            text += f"\n \n"

            logger.info(text)

def load_settings():
    content = np.load(ST_PATH, allow_pickle=True).item()
    settings = {'mode': content['mode'],
                'hsv': 247 if content['seg'] == 'lo' else 245 if content['seg'] == 'mi' else 243,
                'exp': 1.6 if content['seg'] == 'lo' else 1.2 if content['seg'] == 'mi' else 1.0,
                'lr': float(content['lr']),
                'ep': int(content['ep']),
                'pre': int(content['pre'])}
    return settings
