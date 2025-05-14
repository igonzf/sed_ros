import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioData, AudioStamped
import torch
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import torch.nn as nn
import yaml
import torch
import scipy
import numpy as np
from torchaudio.transforms import AmplitudeToDB
from desed_task.nnet.CRNN_e2e import CRNN
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.dataio.datasets_atst_sed import SEDTransform, ATSTTransform, read_audio
from desed_task.utils.scaler import TorchScaler

from collections import OrderedDict
from sound_msgs.msg import SoundEventDetection


classes_labels = OrderedDict(
    {
        "Alarm_bell_ringing": 0,
        "Blender": 1,
        "Cat": 2,
        "Dishes": 3,
        "Dog": 4,
        "Electric_shaver_toothbrush": 5,
        "Frying": 6,
        "Running_water": 7,
        "Speech": 8,
        "Vacuum_cleaner": 9,
    }
)



class ATSTSEDFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sed_feat_extractor = SEDTransform(config["feats"])
        self.scaler = TorchScaler(
                "instance",
                config["scaler"]["normtype"],
                config["scaler"]["dims"],
            )
        self.atst_feat_extractor = ATSTTransform()
    
    def take_log(self, mels):
        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code
    
    def forward(self, mixture):
        mixture = mixture.unsqueeze(0)  # fake batch size
        sed_feats = self.sed_feat_extractor(mixture)
        sed_feats = self.scaler(self.take_log(sed_feats))
        atst_feats = self.atst_feat_extractor(mixture)

        return sed_feats, atst_feats

class ATSTSEDInferencer(nn.Module):
    """Inference module for ATST-SED
    """
    def __init__(
        self, 
        model_config_path="./confs/stage2.yaml", 
        overlap_dur=3, 
        hard_threshold=0.5,
        pretrained_path="./model.pth"):
        super().__init__()

        # Load model configurations
        with open(model_config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config = config

        # Initialize model components but no loading of pretrained weights yet
        self.model = self.load_from_pretrained(pretrained_path, config)

        # Initialize label encoder
        self.label_encoder = ManyHotEncoder(
            list(classes_labels.keys()),
            audio_len=config["data"]["audio_max_len"],
            frame_len=config["feats"]["n_filters"],
            frame_hop=config["feats"]["hop_length"],
            net_pooling=config["data"]["net_subsample"],
            fs=config["data"]["fs"],
        )
        
        # Initialize feature extractor
        self.feature_extractor = ATSTSEDFeatureExtractor(config)
    
        # Initial parameters
        self.audio_dur = 10  # this value is fixed because ATST-SED is trained on 10-second audio
        self.overlap_dur = overlap_dur
        self.fs = config["data"]["fs"]
        
        # Unfolder for splitting audio into chunks
        self.unfolder = nn.Unfold(kernel_size=(self.fs * self.audio_dur, 1), stride=(self.fs * self.overlap_dur, 1))
    
        self.hard_threshold = [hard_threshold] * len(self.label_encoder.labels) if not isinstance(hard_threshold, list) else hard_threshold


    def load_from_pretrained(self, pretrained_path: str, config: dict):

        model = CRNN(
            unfreeze_atst_layer=config["opt"]["tfm_trainable_layers"], 
            **config["net"], 
            model_init=config["ultra"]["model_init"],
            atst_dropout=config["ultra"]["atst_dropout"],
            atst_init=config["ultra"]["atst_init"],
            mode="teacher")
        
        state_dict = torch.load(pretrained_path, map_location="cpu")
        state_dict = {k.replace("sed_teacher.", ""): v for k, v in state_dict.items() if "teacher" in k}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    
    def get_logmel(self, wav_file):
        mixture, onset_s, offset_s, padded_indx = read_audio(
            wav_file, False, False, None
        )
        sed_feats, _ = self.feature_extractor(mixture)
        return sed_feats[0].detach().cpu().numpy()

    def forward(self, wav_file=None, audio_tensor=None):
        if audio_tensor is None:
            mixture, onset_s, offset_s, padded_indx = read_audio(wav_file, False, False, None)
        else:
            mixture = audio_tensor

        # split wav into chunks with overlap
        if (mixture.numel() // self.fs) <= self.audio_dur:
            inference_chunks = [mixture]
            padding_frames = 0
            mixture_pad = mixture.clone()
        else:
            # pad the mixtures
            mixture = mixture.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            total_chunks = (mixture.numel() - ((self.audio_dur - self.overlap_dur) * self.fs)) // (self.overlap_dur * self.fs) + 1
            total_length = total_chunks * self.overlap_dur * self.fs + (self.audio_dur - self.overlap_dur) * self.fs
            mixture_pad = torch.nn.functional.pad(mixture, (0, 0, 0, total_length - mixture.numel()))
            padding_frames = self.time2frame(total_length - mixture.numel())
            inference_chunks = self.unfolder(mixture_pad)
            inference_chunks = inference_chunks.squeeze(0).T
            
        
        # inference result for each chunk
        sed_results = []
        for chunk in inference_chunks:
            sed_feats, atst_feats = self.feature_extractor(chunk)
            chunk_result, _ = self.model(sed_feats, atst_feats)
            sed_results.append(chunk_result)

        if self.hard_threshold is None:
            return sed_results  # If no threshold given, return soft results and you can customize the threshold later
        else:
            chunk_decisions = []
            for i, chunk_result in enumerate(sed_results):
                hard_chunk_result = self.post_process(chunk_result.detach().cpu())
                chunk_decisions.append(hard_chunk_result)
            return self.decision_unify(chunk_decisions, self.time2frame(mixture_pad.numel()), padding_frames)
        
    def post_process(self, strong_preds):
        strong_preds = strong_preds[0]  # only support single input (bsz=1)
        smoothed_preds = []
        for score, fil_val in zip(strong_preds, self.config["training"]["median_window"]):
            score = scipy.ndimage.filters.median_filter(score[:, np.newaxis], (fil_val, 1))
            smoothed_preds.append(score)
        smoothed_preds = np.concatenate(smoothed_preds, axis=1)
        decisions = []
        for score, c_th in zip(smoothed_preds.T, self.hard_threshold):
            pred = score > c_th
            decisions.append(pred[np.newaxis, :] )
        decisions = np.concatenate(decisions, axis=0)
        return decisions
    
    def time2frame(self, time):
        return int(int((time / self.label_encoder.frame_hop)) / self.label_encoder.net_pooling)
    
    def decision_unify(self, chunk_decisions, total_frames, padding_frames):
        C, T = chunk_decisions[0].shape
        if len(chunk_decisions) == 1:
            return chunk_decisions[0]
        else:
            decisions = torch.zeros((C, total_frames), dtype=torch.float32)

            hop_frame = self.time2frame(self.overlap_dur * self.fs)
            for i in range(len(chunk_decisions)):
                chunk_decision_tensor = torch.tensor(chunk_decisions[i], dtype=torch.float32)
                decisions[:, i * hop_frame: i * hop_frame + T] += chunk_decision_tensor

            return (decisions > 0).float()[:, :-padding_frames]


class SoundDetectionNode(Node):

    def __init__(self):
        super().__init__('sound_detection')

        model_config_path = "confs/stage2.yaml"
        pretrained_path = "model/model.pth"
        self.model = ATSTSEDInferencer(model_config_path=model_config_path, pretrained_path=pretrained_path)
        self.model.eval()
        self._pub_sed = self.create_publisher(SoundEventDetection, 'sed', 10)

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT, 
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
            durability=QoSDurabilityPolicy.VOLATILE 
        )
        self._sub_audio = self.create_subscription(
            AudioData,
            '/audio',
            self.audio_callback,
            qos_profile)
        self._sub_audio

    def audio_callback(self, msg):

        audio_tensor = torch.tensor(msg.int16_data, dtype=torch.float32) / 32768.0

        self.get_logger().info('I heard something!')
        sed_results = self.model(audio_tensor)
        
        if sed_results.sum():
            
            sed_results = sed_results * np.arange(1, len(sed_results) + 1).reshape(-1, 1) * 10
            sed_results = np.concatenate([np.zeros_like(sed_results), sed_results], axis=1)
            sed_results = sed_results.reshape(-1, sed_results.shape[1] // 2).astype(float)
            sed_results[sed_results == 0] = np.nan
            sed_results_no_nan = sed_results[~np.isnan(sed_results)]
            mode_value, count = scipy.stats.mode(sed_results_no_nan)
            index = int(mode_value/10-1)
            
            reversed_classes_labels = {v: k for k, v in classes_labels.items()}
            sed = SoundEventDetection()
            sed.header.frame_id = ""
            sed.header.stamp = self.get_clock().now().to_msg()
            sed.class_id = index
            sed.class_name = reversed_classes_labels.get(index, None)
            self._pub_sed.publish(sed)
    
        #else:
            # print("No valid event detected")



def main(args=None):
    rclpy.init(args=args)

    node = SoundDetectionNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()