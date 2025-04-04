import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import AutoModelForTokenClassification, AutoModel, AutoTokenizer, AutoConfig
from gradient_reversal import GradientReversal


def calc_loss_helper(logits, labels, attention_mask, num_even_tags, num_odd_tags):
    # shape: (batch_size, seq_len, num_tags) -> (batch_size, num_tags, seq_len)
    logits = torch.movedim(logits, -1, 1)
    odd_logits, even_logits = torch.split(logits, [num_odd_tags, num_even_tags], dim=1)
    odd_labels = (labels // (num_even_tags + 1)) - 1
    even_labels = (labels % (num_even_tags + 1)) - 1
    # The last word will have only even label

    # Only keep active parts of the loss
    active_even_labels = torch.where(
        attention_mask, even_labels, -1
    )
    active_odd_labels = torch.where(
        attention_mask, odd_labels, -1
    )
    loss = (F.cross_entropy(even_logits, active_even_labels, ignore_index=-1)
            + F.cross_entropy(odd_logits, active_odd_labels, ignore_index=-1))

    return loss


class ModelForTetratagging(nn.Module):
    def __init__(self, config, use_GRL=False,GRL_theta=None,GRL_loss_rate=None,domain_len=None,device='cpu'):
        super().__init__()
        self.num_even_tags = config.task_specific_params['num_even_tags']
        self.num_odd_tags = config.task_specific_params['num_odd_tags']
        self.model_path = config.task_specific_params['model_path']
        self.use_pos = config.task_specific_params.get('use_pos', False)
        self.num_pos_tags = config.task_specific_params.get('num_pos_tags', 50)

        self.pos_emb_dim = config.task_specific_params['pos_emb_dim']
        self.dropout_rate = config.task_specific_params['dropout']

        self.bert = AutoModel.from_pretrained(self.model_path, config=config)
        if self.use_pos:
            self.pos_encoder = nn.Sequential(
                nn.Embedding(self.num_pos_tags, self.pos_emb_dim, padding_idx=0)
            )

        self.endofword_embedding = nn.Embedding(2, self.pos_emb_dim)
        self.lstm = nn.LSTM(
            2 * config.hidden_size + self.pos_emb_dim * (1 + self.use_pos),
            config.hidden_size,
            config.task_specific_params['lstm_layers'],
            dropout=self.dropout_rate,
            batch_first=True, bidirectional=True
        )
        self.projection = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.num_labels)
        )

        self.use_GRL = use_GRL
        if self.use_GRL:
            self.GRL_config = AutoConfig.from_pretrained(self.model_path, config=config)
            self.GRL_tokenizer = AutoTokenizer.from_pretrained(self.model_path, config=config)
            self.GRL_layer = nn.Sequential(GradientReversal(GRL_theta), nn.Linear(self.GRL_config.hidden_size,domain_len))
            self.GRL_loss_rate = GRL_loss_rate
            self.GRL_loss = torch.nn.CrossEntropyLoss()
        self.device = device
        if 'cuda' in self.device:
            self.cuda()

    def GRL_forward(self, words, domains):
        tokenized_word = self.GRL_tokenizer(words, padding=True)
        padding_len = len(tokenized_word['input_ids'][0])
        word_paddings = torch.tensor(tokenized_word['input_ids']).to(self.device)
        attention_mask = torch.tensor(tokenized_word['attention_mask']).to(self.device)
        domains = torch.tensor([[domain] * padding_len for domain in domains]).to(self.device)
        bert_output = self.bert(word_paddings, attention_mask=attention_mask).last_hidden_state
        GRL_output = self.GRL_layer(bert_output)
        # Flatten
        attention_mask = torch.flatten(attention_mask, 0, 1)
        GRL_output_flatten = torch.flatten(GRL_output, 0, 1)[attention_mask]
        domains_flatten = torch.flatten(domains, 0, 1)[attention_mask]
        loss = self.GRL_loss_rate*self.GRL_loss(GRL_output_flatten, domains_flatten)
        return loss

    def forward(
            self,
            input_ids=None,
            pair_ids=None,
            pos_ids=None,
            end_of_word=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if self.use_pos:
            pos_encodings = self.pos_encoder(pos_ids)
            token_repr = torch.cat([outputs[0], pos_encodings], dim=-1)
        else:
            token_repr = outputs[0]

        start_repr = outputs[0].take_along_dim(pair_ids.unsqueeze(-1), dim=1)
        token_repr = torch.cat([token_repr, start_repr], dim=-1)
        token_repr = torch.cat([token_repr, self.endofword_embedding((pos_ids != 0).long())],
                               dim=-1)

        lens = attention_mask.sum(dim=-1).cpu()
        token_repr = pack_padded_sequence(token_repr, lens, batch_first=True,
                                          enforce_sorted=False)
        token_repr = self.lstm(token_repr)[0]
        token_repr, _ = pad_packed_sequence(token_repr, batch_first=True)

        tag_logits = self.projection(token_repr)

        loss = None
        if labels is not None and self.training:
            loss = calc_loss_helper(
                tag_logits, labels, attention_mask.bool(),
                self.num_even_tags, self.num_odd_tags
            )

        return loss, tag_logits
