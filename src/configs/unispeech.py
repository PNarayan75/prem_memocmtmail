# configs/unispeech.py
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
        self.accumulation_steps = 8
        self.learning_rate = 1e-4

        # FIXED: "Spectator" → गलत था!
        self.dropout = 0.3  # ← सही है

        self.loss_type = "CrossEntropyLoss"
        self.checkpoint_dir = "/kaggle/working/checkpoints/IEMOCAP"
        self.model_type = "MemoCMT"

        self.text_encoder_type = "bert"
        self.text_encoder_dim = 768
        self.text_unfreeze = False

        # NEW MODEL
        self.audio_encoder_type = "unispeech-sat-base-plus"
        self.audio_encoder_dim = 768
        self.audio_unfreeze = False

        self.fusion_dim = 384

        # CORRECT DATA ROOT (आपके dataset के अनुसार)
        self.data_root = "/kaggle/working/prem_memocmtmail/IEMOCAP_preprocessed"
        self.data_name = "IEMOCAP"
        self.data_valid = "val.pkl"
        self.text_max_length = 128
        self.audio_max_length = 160000

        self.name = f"{self.model_type}_bert_unispeech"

        for key, value in kwargs.items():
            setattr(self, key, value)
