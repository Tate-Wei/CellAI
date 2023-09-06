import os, logging, copy
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch import optim, nn, Tensor
from torch.utils import data
from typing import Tuple
import torchvision.models as models
import PIL.Image as Image
#from model.src.utils.param import args
import itertools
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.ticker import MaxNLocator

from model.src.utils.utils import Document, Usage, load_settings
from ami.settings import BASE_DIR
from model.src.utils.grad_cam import GradCAM


logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device("mps") # ! Only for mac

class DataLoader():
    def __init__(self, usage: str,
                 dataset: data.Dataset,
                 args_split_rate: float,
                 args_batch_size: int=32,
                 args_num_workers: int=0) -> None:
        
        self.usage = usage
        self.dataset = dataset
        self.batch_size = args_batch_size
        self.num_workers = args_num_workers
        self.split_rate = args_split_rate

        self._init_dataloader()
    
    def _init_dataloader(self) -> None:
        """Create dataloader.
        """

        if self.split_rate:
            assert self.usage != Usage.PREDICTION, "Prediction has split rate."
            tr_size = int(self.split_rate * len(self.dataset))
            val_size = len(self.dataset) - tr_size
            tr_set, val_set = data.random_split(self.dataset, [tr_size, val_size])
            tr_sampler = self._build_sampler(tr_set)

            self.tr_loader = data.DataLoader(tr_set,
                                             batch_size=self.batch_size,
                                             num_workers=self.num_workers,
                                             drop_last=False,
                                             pin_memory=True,
                                             sampler=tr_sampler)
            self.non_weighted_tr_loader = data.DataLoader(tr_set,
                                             batch_size=self.batch_size,
                                             num_workers=self.num_workers,
                                             drop_last=False,
                                             pin_memory=True)
            self.val_loader = data.DataLoader(val_set,
                                              batch_size=self.batch_size,
                                              num_workers=self.num_workers,
                                              drop_last=False,
                                              pin_memory=True)
            self.te_loader = None

        else:
            assert self.usage == Usage.PREDICTION, "Training does not define split rate."
            # Create dataloader for prediction
            self.tr_loader = None
            self.val_loader = None
            self.te_loader = data.DataLoader(self.dataset,
                                             batch_size=self.batch_size,
                                             num_workers=self.num_workers,
                                             shuffle=False,
                                             drop_last=False,
                                             pin_memory=True)
            
    def _build_sampler(self, tr_set) -> data.WeightedRandomSampler:
        """Handle imbalanced dataset.
        Args:
            tr_set: Sub-dataset for training.

        Return:
            Weighted sampler.
        """

        tr_set_idx = tr_set.indices
        labels = [self.dataset.__getitem__(i)['label'] for i in tr_set_idx]
        lb_unique, counts = np.unique(labels, return_counts=True)
        # logger.info(f"Labels: {lb_unique}, {counts}")

        # Compute class weights
        cls_weights = compute_class_weight('balanced',
                                           classes=lb_unique,
                                           y=labels)
        # logger.info(f"Class weights: {cls_weights}")

        # Assign sample weights
        sp_weights = []
        for label in labels:
            for j in range(len(lb_unique)):
                if label == lb_unique[j]:
                    sp_weights.append(cls_weights[j])
                    break
        #logger.info(f"Sample weights: {sp_weights[:5]}, {labels[:5]}") 

        return data.WeightedRandomSampler(sp_weights,
                                          len(sp_weights),
                                          replacement=True)

class AOptimizer(DataLoader):
    def __init__(self,
                 usage: str,
                 dataset: data.Dataset,
                 model: models,
                 optimizer: optim,
                 criterion: nn,
                 args_split_rate: float) -> None:
        
        self.model = model
        self.optim = optimizer
        self.criterion = criterion
        self.epochs = load_settings()['ep']

        super().__init__(usage, dataset, args_split_rate)

    def _prediction(self, output: np.array) -> Tuple[np.array, str]:
        """Compute corresponding label accroding to the model output"""

        with torch.no_grad():
            prob = nn.functional.softmax(output - torch.max(output, 1, True)[0], dim=1).cpu().numpy()
            answer = np.argmax(prob, axis=1)
            return prob, answer

    def _loss_related_batch_loop(self, usage: str, input: Tensor, label: Tensor) -> float:
        """Compute loss"""

        loss = self.criterion(input, label)
        loss = torch.mean(loss, 0)
        if usage == Usage.TRAINING:
            loss.backward()
            self.optim.step()

        elif usage == Usage.VALIDATION:
            pass

        return loss.item()

    def _batch_loop(self, usage: str, dataloader: DataLoader) -> Tuple[list, list, list]:
        """Optimize the parameters in each batch"""

        batch_idx = []
        batch_id = []
        batch_loss = []
        batch_pred = []
        batch_prob = []
        batch_label = []
        mode = load_settings()['mode']
        
        progress_bar = tqdm(dataloader, total=dataloader.__len__())
        for _, batch in enumerate(progress_bar):
            # Bring the data to GPU
            img_idx = batch['idx']
            img_id = batch['id']
            img = batch[mode].to(device)
            label = batch['label'].to(device)

            if usage == Usage.TRAINING:
                self.optim.zero_grad()
                output = self.model(img)
                prob, pred = self._prediction(output)

                loss = self._loss_related_batch_loop(Usage.TRAINING, output, label)
                progress_bar.set_description(f"Avg_loss: {round(loss, 3)}")

            
            elif usage == Usage.VALIDATION:
                # torch.cuda.empty_cache()
                with torch.no_grad():
                    output = self.model(img)
                    prob, pred = self._prediction(output)

                    loss = self._loss_related_batch_loop(Usage.VALIDATION, output, label)
                progress_bar.set_description(f"Avg_loss: {round(loss, 3)}")

            elif usage == Usage.PREDICTION:
                with torch.no_grad():
                    output = self.model(img)
                    prob, pred = self._prediction(output)

                    loss = None

            else:
                raise ValueError(f"Unknown evaluation type: {usage}")
            
            batch_idx.append(img_idx.detach().numpy())
            batch_id.append(img_id)
            batch_loss.append(loss)
            batch_pred.append(pred)
            batch_prob.append(prob)
            batch_label.append(label.cpu().detach().numpy())

        return batch_idx, batch_id, batch_loss, batch_pred, batch_prob, batch_label

    def _enable_dropout(self):

            """ Function to enable the dropout layers during test-time """

            for m in self.model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()

    def _uncertainty_prediction(self, usage: str, dataloader: DataLoader) -> Tuple:
        """Predict the label with uncertainty"""
        if usage == Usage.PREDICTION:

            batch_idx = []
            batch_id = []
            batch_loss = []
            batch_label = []
            forward_passes = 5
            progress_bar = tqdm(dataloader, total=dataloader.__len__())
            dropout_predictions = np.empty((0, len(self.dataset), 5))
            softmax = nn.Softmax(dim=1)
            mode = load_settings()['mode']

            for i in range(forward_passes):

                predictions = np.empty((0, 5))
                self.model.eval()
                self._enable_dropout()

                for _, batch in enumerate(progress_bar):

                    img_idx = batch['idx']
                    img_id = batch['id']
                    img = batch[mode].to(device)
                    #label = batch['label'].to(device)
                    with torch.no_grad():
                        output = self.model(img)
                        output = softmax(output)

                    predictions = np.vstack((predictions, output.cpu().numpy()))

                    if i == 0:

                        batch_idx.append(img_idx.detach().numpy())
                        batch_id.append(img_id)
                        loss = None
                        batch_loss.append(loss)
                        batch_label.append(None)

                dropout_predictions = np.vstack((dropout_predictions, predictions[np.newaxis, :, :]))

            mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)
            batch_pred_ = np.argmax(mean, axis=1) # shape (n_samples,)
            #variance: batch_prob_ = np.mean(np.var(dropout_predictions, axis=0),axis=1)  # shape (n_samples, n_classes)
            batch_prob_ = -np.sum(mean * np.log(mean + 1e-10), axis=1)  # shape (n_samples,)
            batch_idx_ = np.concatenate(batch_idx)
            batch_id_ = sum(batch_id, [])
            avg_loss, conf_mat, f1_sco, accuracy = None, None, None, None
            batch_label_ = None
            batch_loss_ = None

            return {'batch_idx': batch_idx_,
                'batch_id': batch_id_,
                'batch_loss': batch_loss_,
                'batch_pred': batch_pred_,
                'batch_prob': batch_prob_,
                'batch_label': batch_label_,
                'avg_loss': avg_loss,
                'conf_mat': conf_mat,
                'f1_score': f1_sco,
                'accuracy': accuracy}
        
    def _get_heatmap(self,image,index,device):

        model_vgg16_layers = self.model.features[29]

        for param in self.model.features.parameters():
            param.requires_grad = True

        gradcam = GradCAM(self.model, model_vgg16_layers,image,index,device)
        _, heatmap, result, _ = gradcam()
        return heatmap, result

    def _epoch_checkpoint_save(self, epoch: int, file_name: str) -> None:
        """Save the weights for each epoch for later (re-)evaluation"""
        
        if not os.path.exists(BASE_DIR / "model" / "outputs" / "checkpoints"):
            os.makedirs(BASE_DIR / "model" / "outputs" / "checkpoints")

        mode = load_settings()['mode']
        torch.save(self.model.state_dict(),
                   BASE_DIR / "model" / "outputs" / "checkpoints" / f"checkpoint_{mode}.pth")

    def _epoch_conf_mat_save(self, confusion_matrix) -> None:
        """Save the weights for each epoch for later (re-)evaluation"""
        
        classes = ['rbc', 'wbc', 'plt', 'agg', 'oof']
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(5, 5))
        # Plot the confusion matrix as an image
        im = plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        
        plt.colorbar(im,fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45,fontsize=13)
        plt.yticks(tick_marks, classes,fontsize=13)

        # Add labels to each cell
        thresh = confusion_matrix.max() / 2.
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, format(confusion_matrix[i, j], '.2f'),
                        horizontalalignment="center",
                        color="white" if confusion_matrix[i, j] > thresh else "black",fontsize=13)

        # Add axis labels
        plt.ylabel('True label',fontsize=13)
        plt.xlabel('Predicted label',fontsize=13)
        plt.tight_layout()
        

        # Save the figure as an image in the specified directory
        plt.savefig(BASE_DIR / 'static'/ 'plots' / 'confusion_matrix.png')
        plt.close('all')

    def _dataloader_visualizer(self,dl,weighted_dl):    
        labels_list_dl= []
        for i, samples in enumerate(dl):
            labels = samples['label'].view(-1).cpu().tolist()
            labels_list_dl.extend(labels)
        
        labels_dl= np.array(labels_list_dl)

        labels_unique_dl, counts_dl = np.unique(labels_dl, return_counts=True)
        label_map = {'rbc': 0, 'wbc': 1, 'plt': 2, 'agg': 3, 'oof': 4}
        labels_unique_dl = [key for key, value in label_map.items() if value in labels_unique_dl]
        if weighted_dl is None:
            
            plt.bar(labels_unique_dl, counts_dl)
            plt.xlabel('Classes')
            plt.ylabel('Count')
            plt.title('normal_data_loader')
            plt.tight_layout()
            plt.show()

        else:
            labels_list_weighted_dl= []

            for i, samples in enumerate(weighted_dl):
                labels_wdl = samples['label'].view(-1).cpu().tolist()
                labels_list_weighted_dl.extend(labels_wdl)
                
            labels_weighted_dl= np.array(labels_list_weighted_dl)

            labels_unique_weighted_dl, counts_weigthted_dl = np.unique(labels_weighted_dl, return_counts=True)
            labels_unique_weighted_dl = [key for key, value in label_map.items() if value in labels_unique_weighted_dl]
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))

            ax1.bar(labels_unique_dl, counts_dl)
            ax1.set_xlabel('Classes')
            ax1.set_ylabel('Count')
            ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax1.set_title('normal_data_loader')

            ax2.bar(labels_unique_weighted_dl, counts_weigthted_dl)
            ax2.set_xlabel('Classes')
            ax2.set_ylabel('Count')
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.set_title('weighted_data_loader')

            plt.tight_layout()

            plt.savefig(BASE_DIR / 'static'/ 'plots' / 'dataloader_dist.png')
            plt.close('all')

    def train(self) -> None:
        """Train the model and do validation"""

        best_acc, best_ep_acc, best_f1_score, best_ep_f1 = 0, 0, 0, 0
        avg_tr_loss, avg_val_loss, avg_val_acc, avg_val_f1 = [], [], [], []

        for epoch in range(self.epochs):
            logger.info(f"Epoch: \t{epoch}/{self.epochs}")

            # Start training
            logger.info(f"Start training")
            self.model.train()
            self.model.to(device)
            batch_idx, batch_id, batch_loss, batch_pred, batch_prob, batch_label = self._batch_loop(Usage.TRAINING, self.tr_loader) 
            
            # Document loss per epoch
            avg_tr_loss.append(np.array(batch_loss).mean())
            content = {'epoch': epoch,
                       'avg_tr_loss': np.array(batch_loss).mean()}
            Document.docu_training_loss_hist(content=content)

            # Start validation
            logger.info(f"Start validation")
            eval_output = self.eval(Usage.VALIDATION)
            # Document the validation result per epoch
            avg_val_loss.append(eval_output['avg_loss'])
            avg_val_acc.append(eval_output['accuracy'])
            avg_val_f1.append(eval_output['f1_score'].mean())
            content = {'epoch': epoch,
                       'avg_val_loss': eval_output['avg_loss'],
                       'val_acc': eval_output['accuracy'],
                       'val_f1_score': eval_output['f1_score'],
                       'val_conf_mat': eval_output['conf_mat'],
                       'val_prob': eval_output['batch_prob'],
                       'val_label': eval_output['batch_label']}
            Document.docu_eval_hist(content=content, file_name="validation_hist")

            if eval_output['accuracy'] > best_acc:
                best_ep_acc = epoch
                best_acc = eval_output['accuracy']
                best_conf_mat = eval_output['conf_mat']
                self._epoch_checkpoint_save(epoch, 'val_acc')
            if eval_output['f1_score'].mean() > best_f1_score:
                best_ep_f1 = epoch
                best_f1_score = eval_output['f1_score'].mean()
                #self._epoch_checkpoint_save(epoch, 'val_f1')

        # Evaluate folds
        self._epoch_conf_mat_save(eval_output['conf_mat'])
        self._dataloader_visualizer(self.non_weighted_tr_loader,self.tr_loader)
        content = {'best_ep_acc': best_ep_acc,
                   'ep_best_accuracy': best_acc,
                   'best_ep_f1': best_ep_f1,
                   'ep_best_f1_score': best_f1_score}
        Document.docu_eval_hist(content=content, file_name="best_result")

    def eval(self, usage: str) -> dict:
        """Evaluate the training, validation and test sets"""

        self.model.eval()
        if usage == Usage.VALIDATION:
            batch_idx, batch_id, batch_loss, batch_pred, batch_prob, batch_label = self._batch_loop(Usage.VALIDATION, self.val_loader)
            batch_idx_ = np.concatenate(batch_idx)
            batch_id_ = sum(batch_id, [])
            batch_loss_ = np.array(batch_loss)
            batch_pred_ = np.concatenate(batch_pred)
            batch_prob_ = np.concatenate(batch_prob)
            batch_label_ = np.concatenate(batch_label)

            avg_loss = batch_loss_.mean()
            conf_mat = confusion_matrix(batch_pred_, batch_label_)
            f1_sco = f1_score(batch_pred_, batch_label_, average=None)
            accuracy = accuracy_score(batch_pred_, batch_label_)

        elif usage == Usage.PREDICTION:
            batch_idx, batch_id, _, batch_pred, batch_prob, _ = self._batch_loop(Usage.PREDICTION, self.te_loader)
            batch_idx_ = np.concatenate(batch_idx)
            batch_id_ = sum(batch_id, [])
            batch_loss_ = None
            batch_pred_ = np.concatenate(batch_pred)
            batch_prob_ = np.concatenate(batch_prob)
            batch_label_ = None
            avg_loss, conf_mat, f1_sco, accuracy = None, None, None, None

        else:
            raise ValueError(f"Unknown evaluation type: {usage}")
        
        return {'batch_idx': batch_idx_,
                'batch_id': batch_id_,
                'batch_loss': batch_loss_,
                'batch_pred': batch_pred_,
                'batch_prob': batch_prob_,
                'batch_label': batch_label_,
                'avg_loss': avg_loss,
                'conf_mat': conf_mat,
                'f1_score': f1_sco,
                'accuracy': accuracy}

    def prediction(self) -> None:
        """Prediction"""

        logger.info(f"Start prediction")
        self.model.to(device)
        #pred_output = self.eval(Usage.PREDICTION)
        pred_output = self._uncertainty_prediction(Usage.PREDICTION, self.te_loader)
        # Document the prediction result
        content = {'pred_idx': pred_output['batch_idx'],
                   'pred_id': pred_output['batch_id'],
                   'pred_label': pred_output['batch_pred'],
                   'pred_prob': pred_output['batch_prob']}
        Document.docu_eval_pred(content=content, file_name="prediction")

        self._sort_uncertainties(pred_output)

    def _sort_uncertainties(self, pred_output: dict):
        file_path = BASE_DIR / "model" / "inputs" / Usage.A_LEARNING / "temp" / "pr_dataset.npy"
        samples = self.dataset.load_data(file_path)

        mode = load_settings()['mode']
        for i, idx in enumerate(pred_output['batch_idx']):
            samples[idx]['label'] = pred_output['batch_pred'][i]
            samples[idx]['conf'] = pred_output['batch_prob'][i]
            samples[idx]['heatmap'], samples[idx]['heatmap+ori_img'] = self._get_heatmap(Image.fromarray(samples[idx][mode]), samples[idx]['label'], device)

        uncertainties = [samples[idx]['conf'] for idx in samples.keys()]
        sorted_idx = [i for i, v in sorted(enumerate(uncertainties), key=lambda x: x[1], reverse=True)]

        od = OrderedDict()
        for i in sorted_idx:
            od[i] = copy.deepcopy(samples[i])

        self.dataset.save_data(od)

class Optimizer(AOptimizer):
    def __init__(self,
                 usage: str,
                 dataset: data.Dataset,
                 model: models,
                 args_split_rate: float,
                 args_optim: str='SGD') -> None:

        self.lr = load_settings()['lr']
        optimizer = self._init_optimizer(model, args_optim)
        criterion = nn.CrossEntropyLoss()
        
        super().__init__(usage, dataset, model, optimizer, criterion, args_split_rate)
        
        #text = f"Initializing an optimizer:\n"
        #text += f"Optimizer: \t{args_optim}\n"
        #text += f"Learning Rate: \t{self.lr}\n"
        #text += f"Epochs: \t{self.epochs}\n"
        #text += f"Batch Size: \t{self.batch_size}\n\n"
        #logger.info(text)

    def _init_optimizer(self, model: models, args_optim: str) -> optim:
        """Initialize the optimizer"""

        if args_optim == 'SGD':
            return optim.SGD(model.classifier.parameters(),
                             lr=self.lr,
                             momentum=0.9)
        
        elif args_optim == 'Adam':
            return optim.Adam(model.classifier.parameters(), lr=self.lr)
