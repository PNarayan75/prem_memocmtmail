# configs/unispeech.py
from configs.base import Config as BaseConfig

class Config(BaseConfig):
    def add_args(self, **kwargs):
        self.batch_size = 1
        self.num_epochs = 30
        self.accumulation_steps = 8
        self.learning_rate = 1e-4

        self.audio_unfreeze = False   # पहले False रखो
        self.text_unfreeze = False
        self.dropout Spectator = 0.3

        self.model_type = "MemoCMT"
        self.text_encoder_type = "bert"
        self.text_encoder_dim = 768

        # NEW MODEL
        self.audio_encoder_type = "unispeech-sat-base-plus"
        self.audio_encoder_dim = 768
        self.fusion_dim = 384

        self.data_name = "IEMOCAP"
        self.data_root = "/kaggle/working/prem_memocmtmail/IEMOCAP_preprocessed"
        self.text_max_length = 128
        self.audio_max_length = 160000

        self.name = "MemoCMT_bert_unispeech"
