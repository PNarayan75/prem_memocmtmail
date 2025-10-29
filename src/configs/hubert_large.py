# configs/hubert_large.py
from configs.base import Config as BaseConfig

class Config(BaseConfig):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.batch_size = 1
        self.num_epochs = 30
        self.accumulation_steps = 8  # Effective batch_size = 8
        self.learning_rate = 1e-4
        self.dropout = 0.3

        self.loss_type = "CrossEntropyLoss"
        self.checkpoint_dir = "/kaggle/working/checkpoints/IEMOCAP"
        self.model_type = "MemoCMT"

        self.text_encoder_type = "bert"
        self.text_encoder_dim = 768
        self.text_unfreeze = False

        # HUBERT LARGE (NEW!)
        self.audio_encoder_type = "hubert_large"
        self.audio_encoder_dim = 1024  # Large hidden size
        self.audio_unfreeze = False
        self.fusion_dim = 512

        # Dataset
        self.data_root = "/kaggle/working/prem_memocmtmail/IEMOCAP_preprocessed"
        self.data_name = "IEMOCAP"
        self.data_valid = "val.pkl"
        self.text_max_length = 128
        self.audio_max_length = 160000  # 10 sec @16kHz

        self.name = f"{self.model_type}_bert_hubert_large_ft"

        for key, value in kwargs.items():
            setattr(self, key, value)
