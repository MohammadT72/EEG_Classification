from torcheeg.trainers import ClassifierTrainer
from torch import load, unsqueeze, device, cuda, no_grad, sigmoid,from_numpy
from torcheeg.models import FBCNet
import os
import mne
from torcheeg import transforms

pred_trans = transforms.Compose([
                              transforms.BandSignal(),
                              transforms.ToTensor(),
                            ])
class MyPredictor:
    def __init__(self,):
        self.model=None
        self.device = device('cuda') if cuda.is_available() else device('cpu')
    def load_model(self,):
        # Initialize the model
        ckpt_path=os.path.join(os.getcwd(),'model.ckpt')
        model = FBCNet(num_classes=2,
                    num_electrodes=32,
                    chunk_size=128,
                    in_channels=4,
                    num_S=32)

        # Load the checkpoint file
        trainer = ClassifierTrainer.load_from_checkpoint(model=model,
                                    num_classes=2,
                                    lr=1e-5,
                                    weight_decay=1e-4,
                                    accelerator="gpu",checkpoint_path=ckpt_path)
        self.model=trainer.model.to(self.device)
    def predict(self, raw, start_time, end_time):
        # duration=end_time-start_time
        # if duration==0:
        #     raw.crop(tmin=start_time, tmax=start_time+1)
        #     duration=1
        # else:
        #     raw.crop(tmin=start_time, tmax=end_time)

        epochs = mne.make_fixed_length_epochs(raw, duration=1)
        segment=epochs.get_data()[1]
        input_data = pred_trans(eeg=segment)['eeg']
        batch = input_data.float().to(self.device)
        batch = unsqueeze(batch,0)
        self.model.eval()
        with no_grad():
            preds = self.model(batch)
        predicted_labels=sigmoid(preds).argmax(-1)
        return self.get_label(predicted_labels), sigmoid(preds).max().cpu().numpy()
    
    def get_label(self, predictions):
        dict_labels = {
            0 : 'Relax',
            1 : 'Stress'
        }
        return dict_labels[predictions.cpu().item()]