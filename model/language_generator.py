import torch
import torch.nn as nn

class LanguageGenerator(nn.Module):
    def forward(self, prefix_projections, tokens):
        embedding_text = self.embedding_matrix(tokens)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        return self.language_generator(inputs_embed=embedding_cat)

    def greedy_decode(self, prefix_embeddings, device):
        batch_size = prefix_embeddings.shape[0]
        covered_items = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        out = self.language_generator(inputs_embeds=prefix_embeddings, use_cache=True)
        logits, past_key_values = out.logits, out.past_key_values
        next_token = torch.argmax(logits[:, -1, :], dim=-1).reshape((-1, 1))
        sentence_tokens = next_token

        for i in range(self.max_response_length):
            out = self.language_generator(next_token, past_key_values=past_key_values, use_cache=True)
            logits, past_key_values = out.logits, out.past_key_values

            next_token = torch.argmax(logits[:, -1, :], dim=-1).reshape((-1, 1))
            covered_items = torch.logical_or(covered_items, (next_token==0).to(device))
            next_token.masked_fill_(covered_items, 0)
            if(torch.sum(covered_items == False) == 0): break
            sentence_tokens = torch.cat((sentence_tokens, next_token), dim=-1)

        return sentence_tokens

    def __init__(self, language_generator, max_response_length = 40):
        super(LanguageGenerator, self).__init__()
        self.language_generator = language_generator
        self.embedding_size = self.language_generator.transformer.wte.weight.shape[1]
        self.embedding_matrix = self.language_generator.transformer.wte
        self.max_response_length = max_response_length