# src/models/modules.py
import torch
import torch.nn as nn
import torchaudio
from transformers import (
    BertConfig,
    BertModel,
    RobertaConfig,
    RobertaModel,
    FocalNetConfig,
    FocalNetModel,
    UniSpeechSatModel,   # ← यहाँ import करो
)

from configs.base import Config


def build_bert_encoder() -> nn.Module:
    """A function to build bert encoder"""
    config = BertConfig.from_pretrained(
        "bert-base-uncased", output_hidden_states=True, output_attentions=True
    )
    bert = BertModel.from_pretrained("bert-base-uncased", config=config)
    return bert


class HuBertBase(nn.Module):
    def __init__(self, **kwargs):
        super(HuBertBase, self).__init__()
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model()

    def forward(self, x):
        features, _ = self.model(x)
        return features


def build_hubert_base_encoder(cfg: Config) -> nn.Module:
    """A function to build hubert encoder"""
    return HuBertBase()


def build_audio_encoder(cfg: Config) -> nn.Module:
    """A function to build audio encoder"""
    encoder_type = cfg.audio_encoder_type

    if encoder_type == "hubert_base":
        model = build_hubert_base_encoder(cfg)
        dim = 768

    elif encoder_type == "unispeech-sat-base-plus":
        model = UniSpeechSatModel.from_pretrained("microsoft/unispeech-sat-base-plus")
        dim = 768

    elif encoder_type == "hubert_large":
        from transformers import HubertModel
        model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        dim = 1024

    else:
        raise ValueError(f"Invalid audio encoder type: {encoder_type}")

    # Set dim in config
    cfg.audio_encoder_dim = dim
    return model


def build_text_encoder(type: str = "bert") -> nn.Module:
    """A function to build text encoder"""
    encoders = {
        "bert": build_bert_encoder,
    }
    assert type in encoders.keys(), f"Invalid text encoder type: {type}"
    return encoders[type]()
