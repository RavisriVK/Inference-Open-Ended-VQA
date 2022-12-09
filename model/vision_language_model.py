import torch.nn as nn

class VLModel(nn.Module):  
    def forward(self, pixel_values, input_ids, token_type_ids, attention_mask, response_ids):
        multimodal_encoding = self.multimodal_encoder(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        prefix_embeddings = self.mapping_network(multimodal_encoding).view(-1, self.prefix_length, self.language_generator.embedding_size)
        out = self.language_generator(prefix_embeddings, response_ids)
        return out
    
    def generate(self, pixel_values, input_ids, token_type_ids, attention_mask, tokenizer, device):
        multimodal_encoding = self.multimodal_encoder(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        prefix_embeddings = self.mapping_network(multimodal_encoding).view(-1, self.mapping_network.prefix_length, self.language_generator.embedding_size)
        response_tokens = self.language_generator.greedy_decode(prefix_embeddings, device)
        responses = tokenizer.batch_decode(response_tokens, skip_special_tokens=False)
        responses = [response.strip('!').strip() for response in responses]
        return responses
    
    def __init__(self, multimodal_encoder, mapping_network, language_generator):
        super(VLModel, self).__init__()
        self.multimodal_encoder = multimodal_encoder
        self.mapping_network = mapping_network
        self.language_generator = language_generator