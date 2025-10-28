from configs.base import Config as BaseConfig

class Config(BaseConfig):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def add_args(self, **kwargs):
        # self.batch_size = 1
        # self.num_epochs = 3
        self.batch_size = 8
        self.num_epochs = 30  
        self.learning_rate = 1e-4 #new add kiya h
        self.audio_unfreeze = True
        self.text_unfreeze = True
        self.dropout = 0.3
        self.loss_type = "CrossEntropyLoss" 
        self.checkpoint_dir = "/kaggle/working/checkpoints/IEMOCAP"
        self.model_type = "MemoCMT"
        self.text_encoder_type = "bert"
        self.text_encoder_dim = 768
        self.audio_encoder_type = "hubert_base"
        self.audio_encoder_dim = 768
        self.fusion_dim = 768



        # Dataset
        self.data_name = "IEMOCAP"
       
        self.data_root = "/kaggle/working/prem_memocmtmail/IEMOCAP_preprocessed"
        self.data_valid = "val.pkl"
        self.text_max_length = 300
        self.audio_max_length = 1290000

        # Config name
        self.name = f"{self.model_type}_{self.text_encoder_type}_{self.audio_encoder_type}"

        for key, value in kwargs.items():
            setattr(self, key, value)
