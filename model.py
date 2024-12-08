from torch import nn
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
import datetime
from collections import defaultdict, Counter

from utils import Get_subwords_mask_RoBERTa, Get_subwords_mask_BERT, Get_subwords_mask_PhoBERT
from dataset import Dataset
from gradient_reversal import GradientReversal


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True, scale=0):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.scale = scale
        self.weight = nn.Parameter(torch.zeros(n_out, n_in + bias_x, n_in + bias_y))

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        return s.squeeze(1) / self.n_in ** self.scale


class Dependency_Parsing(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Encoding parameters
        self.embedding_type = args.embedding_type
        self.embedding_name = args.embedding_name
        self.embedding_max_len = args.embedding_max_len

        # MLP parameters
        self.arc_mlp = args.arc_mlp
        self.label_mlp = args.label_mlp
        self.drop_out = args.drop_out

        # Training parameters
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.lr_rate = args.lr_rate

        # GRL parameters
        self.use_grl = args.use_grl
        self.grl_theta = args.grl_theta
        self.grl_loss_rate = args.grl_loss_rate
        self.eval_with_grl = args.eval_with_grl

        if args.embedding_type == 'roberta':
            self.get_mask = Get_subwords_mask_RoBERTa
            if 'phobert' in args.embedding_name:
                self.get_mask = Get_subwords_mask_PhoBERT
            self.tokenizer = AutoTokenizer.from_pretrained(args.embedding_name)
            self.encoder_config = AutoConfig.from_pretrained(args.embedding_name)
            self.embedding_max_len = min(self.embedding_max_len, self.encoder_config.max_position_embeddings)
        if args.embedding_type == 'mamba':
            self.get_mask = Get_subwords_mask_RoBERTa
            self.tokenizer = AutoTokenizer.from_pretrained(args.embedding_name)
            self.encoder_config = AutoConfig.from_pretrained(args.embedding_name)
        if args.embedding_type == 'bert':
            self.get_mask = Get_subwords_mask_BERT
            self.tokenizer = AutoTokenizer.from_pretrained(args.embedding_name)
            self.encoder_config = AutoConfig.from_pretrained(args.embedding_name)
        self.device = args.device

        self.train_dataset = self.Data_Preprocess(
            Dataset(directory=args.train_dir, use_folder=args.train_use_folder, use_domain=args.train_use_domain),
            init=True)
        self.dev_dataset = self.Data_Preprocess(
            Dataset(directory=args.dev_dir, use_folder=args.dev_use_folder, use_domain=args.dev_use_domain), init=False)
        self.test_dataset = self.Data_Preprocess(
            Dataset(directory=args.test_dir, use_folder=args.test_use_folder, use_domain=args.test_use_domain),
            init=False)
        self.Build()
        if 'cuda' in self.device:
            self.cuda()

    def Data_Preprocess(self, dataset, init=False):
        if init is True:
            self.pos_tag_vocab = dataset.pos_tag_vocab
            self.label_vocab = dataset.label_vocab
            self.domain_vocab = dataset.domain_vocab
            self.use_domain = dataset.use_domain
        data = []
        for sentence in dataset.data:
            tokenized_words = self.tokenizer.tokenize(' '.join(sentence['words']))
            if len(tokenized_words) + 2 > self.embedding_max_len:
                continue
            origin_masks = self.get_mask(sentence['words'], tokenized_words)
            if self.embedding_type in ['bert', 'roberta']:
                origin_masks = [False] + origin_masks + [False]
            encoded_words = self.tokenizer(' '.join(sentence['words']))['input_ids']
            encoded_heads = sentence['heads']
            encoded_labels = [self.label_vocab[label] for label in sentence['labels']]
            encoded_pos_tags = [self.pos_tag_vocab[pos_tag] for pos_tag in sentence['pos_tags']]
            new_sen = dict(
                {'words': encoded_words, 'heads': encoded_heads, 'labels': encoded_labels, 'pos_tags': encoded_pos_tags,
                 'mask': origin_masks})
            if dataset.use_domain:
                new_sen['domain'] = dataset.domain_vocab[sentence['domain']]
            data.append(new_sen)
        return data

    def Build(self):
        # Encoder layer
        if self.embedding_type in ['bert', 'roberta', 'mamba']:
            self.encoder = AutoModel.from_pretrained(self.embedding_name)
        # MLP layer
        self.head_mlp_arc = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.arc_mlp),
                                          nn.Dropout(self.drop_out), nn.ReLU())
        self.dep_mlp_arc = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.arc_mlp),
                                         nn.Dropout(self.drop_out), nn.ReLU())
        self.head_mlp_label = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.label_mlp),
                                            nn.Dropout(self.drop_out), nn.ReLU())
        self.dep_mlp_label = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.label_mlp),
                                           nn.Dropout(self.drop_out), nn.ReLU())

        # Gradient reversal layer
        if self.use_grl:
            """
            self.GRL = nn.Sequential(GradientReversal(alpha=self.grl_theta),
                                     nn.Linear(self.encoder_config.hidden_size, len(self.domain_vocab)),
                                     nn.ReLU(),
                                     nn.Linear(len(self.domain_vocab), len(self.domain_vocab)),
                                     nn.Softmax())
            """
            self.GRL = nn.Sequential(GradientReversal(alpha=self.grl_theta),
                                     nn.Linear(self.encoder_config.hidden_size, len(self.domain_vocab)),
                                     nn.ReLU())
        # Biaffine layer
        self.biaffine_arc = Biaffine(n_in=self.arc_mlp, n_out=1, bias_x=True, bias_y=False)
        self.biaffine_label = Biaffine(n_in=self.label_mlp, n_out=len(self.label_vocab), bias_x=True, bias_y=True)
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            params=[{'params': p, 'lr': self.learning_rate * (1 if n.startswith('encoder') else self.lr_rate)}
                    for n, p in self.named_parameters()], betas=(0.9, 0.999), lr=self.learning_rate, weight_decay=0)

    def forward(self, data):
        # Split data
        words = [sentence['words'] for sentence in data]
        heads = [sentence['heads'] for sentence in data]
        labels = [sentence['labels'] for sentence in data]
        masks = [sentence['mask'] for sentence in data]
        if self.use_grl:
            domains = [sentence['domain'] for sentence in data]

        # Get length after padding
        n_sentences = len(data)
        max_word_len = max([len(sentence) for sentence in words])
        from torch.nn.functional import pad
        # Padding
        word_paddings = torch.stack([pad(torch.tensor(word), (0, max_word_len - len(word)), value=0) for word in words])
        head_paddings = torch.stack([pad(torch.tensor(head), (0, max_word_len - len(head)), value=0) for head in heads])
        label_paddings = torch.stack(
            [pad(torch.tensor(label), (0, max_word_len - len(label)), value=0) for label in labels])
        mask_paddings = [mask + [False] * (max_word_len - len(mask)) for mask in masks]
        attention_mask = torch.tensor([[1] * len(word) + [0] * (max_word_len - len(word)) for word in words])
        if self.use_grl:
            domain_paddings = torch.tensor([[domain] * max_word_len for domain in domains])

        # Getting contexual embedding
        if self.embedding_type in ['bert', 'roberta', 'mamba']:
            embedding_output = self.encoder(word_paddings, attention_mask=attention_mask).last_hidden_state
            new_embedding_output = torch.stack([torch.cat((embedding[padding],
                                                           torch.zeros(max_word_len - len(embedding[padding]),
                                                                       self.encoder_config.hidden_size))) for
                                                embedding, padding in zip(embedding_output, mask_paddings)])
            embedding_output = new_embedding_output

        # Send the embedding into MLPs
        arc_head = self.head_mlp_arc(embedding_output)
        arc_dep = self.dep_mlp_arc(embedding_output)
        label_head = self.head_mlp_label(embedding_output)
        label_dep = self.dep_mlp_label(embedding_output)
        # Biaffine attention
        arc_scores = self.biaffine_arc(arc_dep, arc_head)
        label_scores = self.biaffine_label(label_dep, label_head)
        label_scores = label_scores.permute(0, 2, 3, 1)

        # GRL transformation
        if self.use_grl:
            domain_scores = self.GRL(embedding_output)

        # Unmask
        mask_paddings = torch.flatten(torch.tensor(mask_paddings))
        unmasked_arc_scores = torch.flatten(arc_scores, 0, 1)[mask_paddings]
        unmasked_label_scores = torch.flatten(label_scores, 0, 1)[mask_paddings]
        unmasked_head_paddings = torch.flatten(head_paddings, 0, 1)[mask_paddings]
        unmasked_label_paddings = torch.flatten(label_paddings, 0, 1)[mask_paddings]
        unmasked_label_scores = unmasked_label_scores[
            [torch.arange(len(unmasked_head_paddings)), unmasked_head_paddings]]

        if self.use_grl:
            unmasked_domain_scores = torch.flatten(domain_scores, 0, 1)[mask_paddings]
            unmasked_domain_paddings = torch.flatten(domain_paddings, 0, 1)[mask_paddings]
        # Calculate loss

        arc_loss = self.loss_fn(unmasked_arc_scores, unmasked_head_paddings)
        label_loss = self.loss_fn(unmasked_label_scores, unmasked_label_paddings)
        loss = arc_loss + label_loss

        if self.use_grl:
            grl_loss = self.loss_fn(unmasked_domain_scores, unmasked_domain_paddings)
            loss = loss + self.grl_loss_rate * grl_loss

        # Get predicted heads & labels
        predicted_heads = unmasked_arc_scores.argmax(1)
        predicted_labels = unmasked_label_scores.argmax(1)

        # Counting correct heads & labels
        correct_heads = int((predicted_heads == unmasked_head_paddings).sum())
        correct_labels = int((predicted_labels == unmasked_label_paddings).sum())
        n_words = mask_paddings.sum()
        # print('uas:', correct_heads/n_words*100)
        # print('las:', correct_labels/n_words*100)

        return loss

    def evaluate(self, data):
        # Turn off gradient
        with torch.no_grad():
            # Split data
            words = [sentence['words'] for sentence in data]
            heads = [sentence['heads'] for sentence in data]
            labels = [sentence['labels'] for sentence in data]
            masks = [sentence['mask'] for sentence in data]
            if self.eval_with_grl:
                domains = [sentence['domain'] for sentence in data]

            # Get length after padding
            n_sentences = len(data)
            max_word_len = max([len(sentence) for sentence in words])
            from torch.nn.functional import pad
            # Padding
            word_paddings = torch.stack(
                [pad(torch.tensor(word), (0, max_word_len - len(word)), value=0) for word in words])
            head_paddings = torch.stack(
                [pad(torch.tensor(head), (0, max_word_len - len(head)), value=0) for head in heads])
            label_paddings = torch.stack(
                [pad(torch.tensor(label), (0, max_word_len - len(label)), value=0) for label in labels])
            mask_paddings = [mask + [False] * (max_word_len - len(mask)) for mask in masks]
            attention_mask = torch.tensor([[1] * len(word) + [0] * (max_word_len - len(word)) for word in words])

            # Getting contexual embedding
            if self.embedding_type in ['bert', 'roberta', 'mamba']:
                embedding_output = self.encoder(word_paddings, attention_mask=attention_mask).last_hidden_state
                new_embedding_output = torch.stack([torch.cat((embedding[padding],
                                                               torch.zeros(max_word_len - len(embedding[padding]),
                                                                           self.encoder_config.hidden_size))) for
                                                    embedding, padding in zip(embedding_output, mask_paddings)])
                embedding_output = new_embedding_output

            # Send the embedding into MLPs
            arc_head = self.head_mlp_arc(embedding_output)
            arc_dep = self.dep_mlp_arc(embedding_output)
            label_head = self.head_mlp_label(embedding_output)
            label_dep = self.dep_mlp_label(embedding_output)
            # Biaffine attention
            arc_scores = self.biaffine_arc(arc_dep, arc_head)
            label_scores = self.biaffine_label(label_dep, label_head).permute(0, 2, 3, 1)
            # Unmask
            mask_paddings = torch.flatten(torch.tensor(mask_paddings))
            unmasked_arc_scores = torch.flatten(arc_scores, 0, 1)[mask_paddings]
            unmasked_label_scores = torch.flatten(label_scores, 0, 1)[mask_paddings]
            unmasked_head_paddings = torch.flatten(head_paddings, 0, 1)[mask_paddings]
            unmasked_label_paddings = torch.flatten(label_paddings, 0, 1)[mask_paddings]

            # Get predicted heads & labels
            predicted_heads = unmasked_arc_scores.argmax(-1)
            unmasked_label_scores = unmasked_label_scores[[torch.arange(len(predicted_heads)), predicted_heads]]
            predicted_labels = unmasked_label_scores.argmax(-1)

            # Counting correct heads & labels
            correct_heads = int((predicted_heads == unmasked_head_paddings).sum())
            correct_labels = int(
                ((predicted_labels == unmasked_label_paddings) & (predicted_heads == unmasked_head_paddings)).sum())

            n_words = int(mask_paddings.sum())
            return (n_words, correct_heads, correct_labels)

    def Train(self, n_epochs, logger, save_dir=None):
        # If save_dir is None, model will not be saved.
        n_batches = (len(self.train_dataset) + self.batch_size - 1) // self.batch_size
        logger.info(f'Number of batches: {n_batches}')
        best_dev_uas, best_dev_las, best_test_uas, best_test_las = 0, 0, 0, 0
        best_epoch = 0
        for epoch_id in range(n_epochs):
            self.train()
            stats = Counter()
            start_time = datetime.datetime.now()
            np.random.shuffle(self.train_dataset)
            for batch in range(0, len(self.train_dataset), self.batch_size):
                data = self.train_dataset[batch:min(batch + self.batch_size, len(self.train_dataset))]
                loss = self(data)
                stats['train_loss'] += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            avg_loss = stats['train_loss'] / n_batches
            logger.info(f'Epoch {epoch_id + 1}: {avg_loss}, {datetime.datetime.now() - start_time} seconds.')
            dev_uas, dev_las = self.Eval(self.dev_dataset)
            test_uas, test_las = self.Eval(self.test_dataset)
            logger.info(f'Dev  set:\tUAS: {round(dev_uas, 2)}\tLAS: {round(dev_las, 2)}')
            logger.info(f'Test set:\tUAS: {round(test_uas, 2)}\tLAS: {round(test_las, 2)}')
            if dev_las > best_dev_las:
                best_dev_uas = dev_uas
                best_dev_las = dev_las
                best_test_uas = test_uas
                best_test_las = test_las
                best_epoch = epoch_id + 1
                logger.info('New best record is saved.')
                if save_dir is not None:
                    torch.save(self.state_dict(), save_dir)
        logger.info(f'Best record on epoch {best_epoch}:')
        logger.info(f'Dev  set:\tUAS: {round(best_dev_uas, 2)}\tLAS: {round(best_dev_las, 2)}')
        logger.info(f'Test set:\tUAS: {round(best_test_uas, 2)}\tLAS: {round(best_test_las, 2)}')

    def Eval(self, dataset, require_preprocessing=False, logger=None):
        self.eval()
        if require_preprocessing:
            eval_data = self.Data_Preprocess(dataset, init=False)
        else:
            eval_data = dataset
        n_batches = (len(eval_data) + self.batch_size - 1) // self.batch_size
        if logger is not None:
            logger.info('Starting evaluation process:')
            logger.info(f'Number of batches: {n_batches}')
        records = defaultdict(int)
        for batch in range(0, len(eval_data), self.batch_size):
            start_time = datetime.datetime.now()
            data = eval_data[batch:min(batch + self.batch_size, len(eval_data))]
            n_words, correct_heads, correct_labels = self.evaluate(data)
            records['words'] += n_words
            records['correct_heads'] += correct_heads
            records['correct_labels'] += correct_labels
        uas = records['correct_heads'] / records['words'] * 100
        las = records['correct_labels'] / records['words'] * 100
        if logger is not None:
            logger.info(f'UAS: {round(uas, 2)}')
            logger.info(f'LAS: {round(las, 2)}')
        return uas, las
