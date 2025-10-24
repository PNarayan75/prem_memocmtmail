from configs.base import Config as BaseConfig

class Config(BaseConfig):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.batch_size = 1
        self.num_epochs = 1

        self.loss_type = "CrossEntropyLoss"

       
        self.checkpoint_dir = "/kaggle/working/checkpoints/IEMOCAP"

        self.model_type = "MemoCMT"

        self.text_encoder_type = "bert"
        self.text_encoder_dim = 768
        self.text_unfreeze = False

        self.audio_encoder_type = "hubert_base"
        self.audio_encoder_dim = 768
        self.audio_unfreeze = False

        self.fusion_dim = 768

        # Dataset
        self.data_name = "IEMOCAP"
       
        self.data_root = "/kaggle/working/prem_memocmtmail/IEMOCAP_preprocessed"
        self.data_valid = "val.pkl"
        self.text_max_length = 297
        self.audio_max_length = 128000

        # Config name
        self.name = f"{self.model_type}_{self.text_encoder_type}_{self.audio_encoder_type}"

        for key, value in kwargs.items():
            setattr(self, key, value)
