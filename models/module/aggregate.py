import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

from models.module.cross_attention import CrossAttentionBlock


class MLP(nn.Module):
    def __init__(self, mlp_hidden_size, hidden_size):
        super().__init__()
        if mlp_hidden_size == hidden_size:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(mlp_hidden_size, hidden_size)

        self.mlp = nn.Sequential(
            nn.LayerNorm(mlp_hidden_size),
            nn.Linear(mlp_hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        return self.skip(x) + self.mlp(x)


class Aggregator(nn.Module):
    def __init__(self, config, question_encoder, baselm_token_dim, amort_enc_dim,
                 num_virtual_tokens, num_actual_tokens, dropout_p=0.):
        super().__init__()
        self.config = config
        self.num_virtual_tokens = num_virtual_tokens
        self.question_encoder = question_encoder
        self.mlps = nn.ModuleList([
            MLP(amort_enc_dim, baselm_token_dim)
            for _ in range(num_virtual_tokens)
        ])

        num_cross_attention_blocks = self.config.num_cross_attention_blocks
        self.cross_attention_list = nn.ModuleList([
            CrossAttentionBlock(dim=baselm_token_dim, context_dim=baselm_token_dim,
                                proj_drop=dropout_p, attn_drop=dropout_p,
                                num_heads=8, dim_head=96)
            for i in range(num_cross_attention_blocks)]
        )

    def forward(self, question_indices, qa_attention, prompt_set):
        # prompt set into batch of prompt set
        batch_prompt_set = prompt_set.unsqueeze(0).repeat(
            question_indices.shape[0], *([1] * len(prompt_set.shape))
        )
        # batch_prompt_set = rearrange(batch_prompt_set, 'b s n d -> b (s n) d')

        hidden_state = self.question_encoder(
            input_ids=question_indices,
            attention_mask=qa_attention,
        )
        # print(question_indices.shape)
        # print(hidden_state.shape) 768? -> mlps for virtual tokenization
        # print(batch_prompt_set.shape) 1280?

        question_token = []
        for i, mlp in enumerate(self.mlps):
            question_token.append(mlp(hidden_state[:, i:i + 1, :]))

        question_token = torch.cat(question_token, dim=1)

        question_token = question_token.unsqueeze(1) # b x 1 x n x d
        question_token_norm = question_token / question_token.norm(dim=3, keepdim=True) # b x 1 x n x d
        batch_prompt_set_norm = batch_prompt_set / batch_prompt_set.norm(dim=3, keepdim=True) # b x s x n x d

        cos_sim = (question_token_norm * batch_prompt_set_norm).sum(dim=-1) # b x s x n
        cos_sim_prob = F.softmax(cos_sim, dim=1) # b x s x n
        
        if self.config.agg_option == 'document-wise':
            cos_sim_prob_sum = cos_sim_prob.sum(dim=-1) # b x s 
            _, top_1_indices = cos_sim_prob_sum.max(dim=1) # b. indices about s
            result = batch_prompt_set[torch.arange(top_1_indices.shape[0]), top_1_indices] # b x n x d, 2 x 24 x 1280
            print(f"s={cos_sim_prob_sum.shape[1]}") 

        else: # token-wise, not implemented yet
            _, top_1_indices = cos_sim_prob.max(dim=1) # b x n. indices about s
            result = None
        
        return result # b x n x d


class PassPrompt(nn.Module):
    def __init__(self, config, question_encoder, hidden_size, mlp_hidden_size,
                 num_virtual_tokens, num_actual_tokens, dropout_p=0.):
        super().__init__()

    def forward(self, question_indices, qa_attention, prompt_set):
        batch_prompt_set = prompt_set.unsqueeze(0).repeat(
            question_indices.shape[0], *([1] * len(prompt_set.shape))
        )
        batch_prompt_set = rearrange(batch_prompt_set, 'b s n d -> b (s n) d')
        return batch_prompt_set
        # return prompt_set
