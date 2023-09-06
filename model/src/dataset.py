import os, logging, h5py, cv2, copy
import numpy as np
import torch
from typing import Tuple, Optional, Callable
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

#from src.utils.param import args
from model.src.utils.utils import Usage, load_settings
from model.src.utils.colormap import CellfaceStdCMap
from ami.settings import BASE_DIR

logger = logging.getLogger(__name__)

NUM2STR = {0: 'rbc', 1: 'wbc', 2: 'plt', 3: 'agg', 4: 'oof'}
STR2NUM = {'rbc': 0, 'wbc': 1, 'plt': 2, 'agg': 3, 'oof': 4}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, usage: str, transform: Optional[Callable]=None) -> None:
        """
        Args:
            usage: Load dataset for segmetation/classification, training/
                   validation/testing/prediction.

        Returns:
            float: Remaining Money
        """

        assert usage in (Usage.TRAINING, Usage.A_LEARNING, Usage.PREDICTION), "Unknown usage."

        self.transform = transform
        self.usage = usage
        if self.usage == Usage.TRAINING:
            raw_data_path = BASE_DIR / "model" / "inputs" / Usage.TRAINING
        elif self.usage == Usage.PREDICTION:
            raw_data_path = BASE_DIR / "model" / "inputs" / Usage.PREDICTION
        else:
            raw_data_path = BASE_DIR / "model" / "inputs" / Usage.A_LEARNING

        logger.info(f"Loading dataset in {raw_data_path} for {self.usage}")

        # Get raw data
        if self.usage == Usage.TRAINING or self.usage == Usage.PREDICTION:
            self.data = self._fetch_data(raw_data_path)
            self._preprocess_data()
        elif self.usage == Usage.A_LEARNING:
            self.data = self._join_data(raw_data_path)
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get sample via DataLoader.
        Args:
            idx: The sample index.

        Returns:
            dict: 'id' for image id, 'img' for amplitude and phase image, 'mask'
                  for mask that fits for original image size, 'bbox' for 
                  bounding box (x, y, w, h) that fits for original image size,
                  'label' for ground truth (None for prediction).
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        keys_to_keep = ['idx', 'id', 'amp', 'ph', 'label']
        filtered_dict = {key: self.samples[idx][key] for key in keys_to_keep if key in self.samples[idx]}
        sample = copy.deepcopy(filtered_dict)

        if self.transform:
            sample['ph'] = self.transform(Image.fromarray(sample['ph']))
            sample['amp'] = self.transform(Image.fromarray(sample['amp']))
        
        return sample
    
    def _join_data(self, raw_data_path: str) -> dict:
        
        # Add new label
        file_path = os.path.join(raw_data_path, "temp", "pr_dataset.npy")
        old_pred = self.load_data(file_path)
        file_path = os.path.join(raw_data_path, "temp", "al_label.npy")
        new_label = self.load_data(file_path)

        new_dataset = {}
        for key in new_label.keys():
            old_pred[int(key)]['label'] = new_label[int(key)]['label']
            old_pred[int(key)]['conf'] = 1
            new_dataset[int(key)] = copy.deepcopy(old_pred[int(key)])
        self.save_data(new_dataset)

        # Join available datset
        file_path = os.path.join(raw_data_path, "dataset")
        ava_datasets = os.listdir(file_path)
        datasets_len = 0
        datasets = []
        for subset in ava_datasets:
            if subset[0] == '.':
                continue
            subset_path = os.path.join(file_path, subset)
            datasets.append(self.load_data(subset_path))
            datasets_len += len(datasets[-1])

        samples = {}
        i = 0
        for subset in datasets:
            for key in subset.keys():
                samples[i] = subset[int(key)]
                samples[i]['idx'] = i
                i += 1

        assert len(samples) == datasets_len, "DEBUG: datasets_len do not match"
        self.samples = copy.deepcopy(samples)

        return self.samples
    
    def _fetch_data(self, raw_data_path: str):
        
        """Fetch the raw data.
        Args:
            raw_data_path: The directory path of available files.

        Returns:
            tuple: Tuple of file name array, amplitude array, phase array, 
                   mask array, label array. For some usage some of them could 
                   be None.
        """

        raw_data_files = os.listdir(raw_data_path)
        logger.info(f"The available files: {raw_data_files}")
        img_id, amplitude, phase, mask, label = [], [], [], [], []

        for file_name in raw_data_files:
            if file_name[0] == '.':
                continue
            file_path = os.path.join(raw_data_path, file_name)
            file = h5py.File(file_path, "r")

            if self.usage == Usage.TRAINING:
                # Preform training, validation.
                img_id.append(file_name)
                amplitude.append(np.array(file['amplitude']['images'])[None, :])
                phase.append(np.array(file['phase']['images'])[None, :])
                label.append(np.array(file['label']['ground_truth'])[None, :])
                try:
                    mask.append(np.array(file['mask']['images'])[None, :])
                except KeyError:
                    pass
            elif self.usage == Usage.PREDICTION:
                # Preform prediction
                pred_num = load_settings()['pre']

                img_id.append(file_name)
                amplitude.append(np.array(file['amplitude']['images'])[:pred_num, ...][None, :])
                phase.append(np.array(file['phase']['images'])[:pred_num, ...][None, :])
            else:
                raise NotImplementedError

        try:
            img_id_array = np.array(img_id, dtype='object')
            amplitude_array = np.concatenate(amplitude, axis=0)
            phase_array = np.concatenate(phase, axis=0)
            mask_array = np.concatenate(mask, axis=0) if mask else None
            label_array = np.concatenate(label, axis=0) if label else None
        except ValueError:
            raise ValueError(f"Opening {file_path}, the input array dimensions do not match.")

        amplitude_array = np.nan_to_num(amplitude_array, nan=0)
        phase_array = np.nan_to_num(phase_array, nan=0)
        mask_array = np.nan_to_num(mask_array, nan=0) if mask else None

        return [img_id_array, amplitude_array, phase_array, mask_array,
                label_array, None, None]
    
    def _get_single_img(self,
                        img_size: int=96,
                        bin_threshold: float=10,
                        amp_background: float=0,
                        ph_background: float=0):
        """Using mask or predicted mask filter background and get single image.
        Args:
            img_size: Target image size.
            bin_threshold: Threshold to create binary image.
            expanding: Factor to expand the mask.
            amp_background: Background value for amplitude.
            ph_background: Background value for phase.

        Returns:
            sub_id: Numpy array for single image ID.
            sub_amp: Numpy array for single amplitude images.
            sub_ph: Numpy array for single phase images.
            sub_mk: Numpy array for mask that fits original size.
            sub_lb: Numpy array for label. None for prediction.
        """

        img_id_array = self.data[0]
        amplitude_array = self.data[1]
        phase_array = self.data[2]
        mask_array = self.data[3]
        label_array = self.data[4]

        # amp4mk_array = amplitude_array.copy()
        # amp4mk_array[amp4mk_array <= -3] = 255
        # amp4mk_array[amp4mk_array >= 3] = 255
        # amp4mk_array[amp4mk_array <= 0] = 0
        # amp4mk_array = np.array(amp4mk_array, dtype=np.uint8)

        hsv_param = load_settings()['hsv']
        expanding = load_settings()['exp']

        ph_array = np.squeeze(np.array(phase_array).astype(np.float32))
        colormap_array = plt.get_cmap(CellfaceStdCMap)(ph_array[:,:,:])
        array_uint8 = (colormap_array * 255).astype(np.uint8)[:,:,:,:3]
        ph4mk_array = np.empty_like(array_uint8[:,:,:,0])
        for i in range(array_uint8.shape[0]):
            bgr_img = cv2.cvtColor(array_uint8[i], cv2.COLOR_RGB2BGR)
            hsv_img= cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
            single_amp4mk_array = cv2.inRange(hsv_img,(0, 0, 0), (255, 255, hsv_param))
            ph4mk_array[i] = single_amp4mk_array
        ph4mk_array = np.array(ph4mk_array, dtype=np.uint8)
        ph4mk_array = ph4mk_array[np.newaxis, ...]


        sub_amp, sub_ph, sub_ori_amp, sub_ori_ph = [], [], [], []
        sub_id, sub_lb = [], []
        for j in range(amplitude_array.shape[0]):
            for i in range(amplitude_array[j].shape[0]):
                ph4mk = np.pad(ph4mk_array[j, i, ...], int(img_size / 2), 
                                mode='constant', constant_values=amp_background)
                amplitude = np.pad(amplitude_array[j, i, ...], int(img_size / 2), 
                                   mode='constant', constant_values=amp_background)
                phase = np.pad(phase_array[j, i, ...], int(img_size / 2), 
                               mode='constant', constant_values=ph_background)
                
                if self.usage == Usage.PREDICTION:
                    # Use predicted mask via OpenCV
                    _, binary = cv2.threshold(ph4mk, bin_threshold, 255, 
                                              cv2.THRESH_BINARY)
                else:
                    # Use true mask
                    binary = np.pad(mask_array[j, i, ...], int(img_size / 2), 
                                    mode='constant')
                
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_SIMPLE)
                
                mk_id = 0
                for contour in contours:
                    mom = cv2.moments(contour)
                    try:
                        center = np.array([int(mom["m10"] / mom["m00"]),
                                           int(mom["m01"] / mom["m00"])])
                        if cv2.contourArea(contour) <= 50:
                            continue
                    except:
                        continue

                    mask = np.zeros(amplitude.shape, dtype=np.uint8)
                    contour = np.array((contour - center) * expanding + center, 
                                        dtype='int32')
                    cv2.drawContours(mask, [contour], -1, 1, cv2.FILLED)

                    single_amp = amplitude * mask
                    single_ph = phase * mask
                    single_amp = single_amp[center[1] - int(img_size / 2):center[1] + int(img_size / 2),
                                            center[0] - int(img_size / 2):center[0] + int(img_size / 2)]
                    ori_amp = amplitude[center[1] - int(img_size / 2):center[1] + int(img_size / 2),
                                        center[0] - int(img_size / 2):center[0] + int(img_size / 2)]
                    single_ph = single_ph[center[1] - int(img_size / 2):center[1] + int(img_size / 2),
                                          center[0] - int(img_size / 2):center[0] + int(img_size / 2)]
                    ori_ph = phase[center[1] - int(img_size / 2):center[1] + int(img_size / 2),
                                   center[0] - int(img_size / 2):center[0] + int(img_size / 2)]
                    
                    sub_id.append(np.array([img_id_array[j], f'img_{i}', f'mk_{mk_id}'], dtype='object')[None, :])
                    sub_amp.append(single_amp[None, :])
                    sub_ori_amp.append(ori_amp[None, :])
                    sub_ph.append(single_ph[None, :])
                    sub_ori_ph.append(ori_ph[None, :])
                    if self.usage != Usage.PREDICTION:
                        sub_lb.append(label_array[j, i].decode('utf-8'))

        sub_id = np.concatenate(sub_id, axis=0)
        sub_amp = np.concatenate(sub_amp, axis=0)
        sub_ori_amp = np.concatenate(sub_ori_amp, axis=0)
        sub_ph = np.concatenate(sub_ph, axis=0)
        sub_ori_ph = np.concatenate(sub_ori_ph, axis=0)
        sub_lb = np.array(sub_lb, dtype='object') if sub_lb else None

        self.data[0] = sub_id
        self.data[1] = sub_amp
        self.data[2] = sub_ph
        self.data[3] = None # Mask is then useless
        self.data[4] = sub_lb
        self.data[5] = sub_ori_amp
        self.data[6] = sub_ori_ph
        return sub_id, sub_amp, sub_ph, None, sub_lb, sub_ori_amp, sub_ori_ph
 
    def _preprocess_data(self) -> None:
        """Preprocess the raw data"""

        self._get_single_img()
        self._colorize_img()
        self._map_label()
        self._package_data()
        self.save_data(self.samples)

    def _colorize_img(self,
                      amp_lo: float=15,
                      amp_hi: float=255,
                      ph_lo: float=2,
                      ph_hi: float=2*np.pi) -> Tuple[np.array, np.array]:
        """Colorize and normalize amplitude and phase images.
        Args:
            amp_lo: Lower bound for amplitude.
            amp_hi: Upper bound for amplitude.
            ph_lo: Lower bound for phase.
            ph_hi: Upper bound for phase.

        Returns:
            amplitude_array: Normalized Numpy array.
            phase_array: Normalized Numpy array.
        """

        amplitude_array = self.data[1]
        ori_amp_array = self.data[5]
        phase_array = self.data[2]
        ori_ph_array = self.data[6]

        amplitude_array = np.clip(amplitude_array, -15, 255)
        amplitude_array = (amplitude_array + 15) / (15 + 255)

        amp_arr_list, ori_amp_arr_list = [], []
        ph_arr_list, ori_ph_arr_list = [], []
        for i in range(amplitude_array.shape[0]):
            co_amp_array = plt.get_cmap('prism')(amplitude_array[i, ...])
            co_amp_array = (co_amp_array * 255).astype(np.uint8)[:, :, :3]
            amp_arr_list.append(co_amp_array[None, :])
            #co_amp_array = plt.get_cmap('prism')(ori_amp_array[i, ...])
            #co_amp_array = (co_amp_array * 255).astype(np.uint8)[:, :, :3]
            #ori_amp_arr_list.append(co_amp_array[None, :])

            co_ph_array = plt.get_cmap(CellfaceStdCMap)(phase_array[i, ...])
            co_ph_array = (co_ph_array * 255).astype(np.uint8)[:, :, :3]
            ph_arr_list.append(co_ph_array[None, :])
            #co_ph_array = plt.get_cmap(CellfaceStdCMap)(ori_ph_array[i, ...])
            #co_ph_array = (co_ph_array * 255).astype(np.uint8)[:, :, :3]
            #ori_ph_arr_list.append(co_ph_array[None, :])

        amplitude_array = np.concatenate(amp_arr_list, axis=0)
        #ori_amp_array = np.concatenate(ori_amp_arr_list, axis=0)
        phase_array = np.concatenate(ph_arr_list, axis=0)
        #ori_ph_array = np.concatenate(ori_ph_arr_list, axis=0)

        self.data[1] = amplitude_array
        #self.data[5] = ori_amp_array
        self.data[2] = phase_array
        #self.data[6] = ori_ph_array
        return amplitude_array, phase_array, ori_amp_array, ori_ph_array
    
    def _map_label(self) -> np.array:
        """Map the label to number.
        Args:

        Returns:
            label: Numpy array containts encoded labels.
        """

        if self.data[4] is None:
            return None
        
        else:
            label = []
            for lb in self.data[4]:
                label.append(STR2NUM[lb])

            self.data[4] = np.array(label)
            return self.data[4]
        
    def save_data(self, file) -> None:

        if self.usage == Usage.TRAINING or self.usage == Usage.A_LEARNING:
            file_path = BASE_DIR / "model" / "inputs" / Usage.A_LEARNING / "dataset"
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            time_now = datetime.now()
            current_time = time_now.strftime("%H_%M_%S")
            file_path = os.path.join(file_path, f"al_dataset_{current_time}.npy")

        elif self.usage == Usage.PREDICTION:
            file_path = BASE_DIR / "model" / "inputs" / Usage.A_LEARNING / "temp"
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            file_path = os.path.join(file_path, "pr_dataset.npy")

        else:
            raise NotImplementedError

        np.save(file_path, file)

    def load_data(self, file_path: str) -> dict:

        assert os.path.exists(file_path), "Load empty dataset."
        return np.load(file_path, allow_pickle=True).item()

    def _package_data(self) -> dict:
        """Colorize and normalize amplitude and phase images.
        Args:
            amp_lo: Lower bound for amplitude.
            amp_hi: Upper bound for amplitude.
            ph_lo: Lower bound for phase.
            ph_hi: Upper bound for phase.

        Returns:
            amplitude_array: Normalized Numpy array.
            phase_array: Normalized Numpy array.
        """
        
        if self.data[4] is not None:
            assert self.data[0].shape[0] \
                == self.data[1].shape[0] \
                == self.data[2].shape[0] \
                == self.data[4].shape[0] \
                == self.data[5].shape[0] \
                == self.data[6].shape[0], "Data size doesn't match."
            conf = np.ones(self.data[1].shape[0], dtype=float)
        else:
            assert self.data[0].shape[0] \
                == self.data[1].shape[0] \
                == self.data[2].shape[0] \
                == self.data[5].shape[0] \
                == self.data[6].shape[0], "Data size doesn't match."
            self.data[4] = np.ones(self.data[1].shape[0], dtype=np.int64) * 9
            conf = np.zeros(self.data[1].shape[0], dtype=float)

        self.samples = {}
        for i in range(self.data[0].shape[0]):
            sample = {
                i: {'idx': i,
                    'id': '+'.join(self.data[0][i]),
                    'amp': self.data[1][i],
                    'ph': self.data[2][i],
                    'ori_amp': self.data[5][i],
                    'ori_ph': self.data[6][i],
                    'label': self.data[4][i],
                    'conf': conf[i]}
                }
            self.samples.update(sample)

        ########################################################################
        """for i in self.samples.keys():
            img = self.samples[i]['img']
            plt.figure()
            plt.imshow(img)
            pass"""
        ########################################################################

        return self.samples


