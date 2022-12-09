import torch.nn as nn

class MultimodalEncoder(nn.Module):
    def __init__(self, multimodal_encoder):
        super(MultimodalEncoder, self).__init__()
        self.multimodal_encoder = multimodal_encoder
    
    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values):
        multimodal_encoding = self.multimodal_encoder(
            input_ids= input_ids,
            token_type_ids= token_type_ids,
            attention_mask= attention_mask,
            pixel_values= pixel_values
        ).multimodal_embeddings
        return multimodal_encoding