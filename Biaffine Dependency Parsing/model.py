from torch import nn
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_scheduler, RobertaTokenizerFast
import transformers
import numpy as np
import datetime
from collections import defaultdict, Counter
import json
import copy
from utils import Get_subwords_mask_RoBERTa, Get_subwords_mask_BERT, Get_subwords_mask_PhoBERT
from dataset import Dataset
from gradient_reversal import GradientReversal
from optim import LinearLR
from pretrained import TransformerEmbedding

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
        self.grl_drop_out = args.grl_drop_out
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
        self.model_save_dir = f'{args.save_dir}/{args.model_name}.bin'
        self.config_save_dir = f'{args.save_dir}/{args.model_name}.config.json'
        self.warm_up_rate = args.warm_up_rate
        if args.mode == 'train':
            self.train_dataset = self.Data_Preprocess(
                Dataset(directory=args.train_dir, use_folder=args.train_use_folder, use_domain=args.train_use_domain),
                init=True)
            self.dev_dataset = self.Data_Preprocess(
                Dataset(directory=args.dev_dir, use_folder=args.dev_use_folder, use_domain=args.dev_use_domain),
                init=False)
            self.test_dataset = self.Data_Preprocess(
                Dataset(directory=args.test_dir, use_folder=args.test_use_folder, use_domain=args.test_use_domain),
                init=False)
            self.num_training_steps = args.n_epochs * (
                    (len(self.train_dataset) + self.batch_size - 1) // self.batch_size)
        if args.mode in ['evaluate', 'predict']:
            config = json.load(open(self.config_save_dir))
            self.pos_tag_vocab = config['pos_tag_vocab']
            self.label_vocab = config['label_vocab']
            self.domain_vocab = config['domain_vocab']
            self.num_training_steps = 10 ** 9
        self.Build()
        if args.mode in ['evaluate', 'predict']:
            self.load_state_dict(torch.load(self.model_save_dir, weights_only=False), strict=False)
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
            original_sentence = ' '.join(sentence['words'])
            tokens = self.tokenizer(original_sentence)['input_ids'][1:-1]
            mask = self.get_mask(sentence['words'],self.tokenizer.tokenize(original_sentence))
            encoded_words = [[0]]
            current_word = []
            for i in range(len(tokens)):
                current_word.append(tokens[i])
                if mask[i] == True:
                    encoded_words.append(current_word)
                    current_word = []
            encoded_heads = sentence['heads']
            encoded_labels = [self.label_vocab[label] for label in sentence['labels']]
            encoded_pos_tags = [self.pos_tag_vocab[pos_tag] for pos_tag in sentence['pos_tags']]
            new_sen = dict(
                {'words': encoded_words, 'heads': encoded_heads, 'labels': encoded_labels, 'pos_tags': encoded_pos_tags})
            if dataset.use_domain:
                new_sen['domain'] = dataset.domain_vocab[sentence['domain']]
            data.append(new_sen)
        return data

    def Build(self):
        # Encoder layer
        if self.embedding_type in ['bert', 'roberta', 'mamba']:
            #self.encoder = AutoModel.from_pretrained(self.embedding_name)
            self.encoder = TransformerEmbedding(name = self.embedding_name, n_layers = 4, pad_index = 1, finetune = True)
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
                                     nn.Dropout(self.grl_drop_out), nn.ReLU())
        # Biaffine layer
        self.biaffine_arc = Biaffine(n_in=self.arc_mlp, n_out=1, bias_x=True, bias_y=False)
        self.biaffine_label = Biaffine(n_in=self.label_mlp, n_out=len(self.label_vocab), bias_x=True, bias_y=True)
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            params=[{'params': p, 'lr': self.learning_rate * (1 if n.startswith('encoder') else self.lr_rate)}
                    for n, p in self.named_parameters()], betas=(0.9, 0.999), lr=self.learning_rate, weight_decay=0)
        '''
        self.scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=int(self.num_training_steps * self.warm_up_rate),
            num_training_steps=self.num_training_steps,
            lr_end=self.learning_rate/10,
            power=0.7,
            last_epoch=-1)
        self.scheduler = get_scheduler(name="linear", optimizer=self.optimizer, num_warmup_steps=int(self.num_training_steps * self.warm_up_rate),
                                       num_training_steps=self.num_training_steps)
        '''
        self.scheduler = LinearLR(optimizer=self.optimizer,
                                  warmup_steps=int(self.num_training_steps * self.warm_up_rate),
                                  steps=self.num_training_steps)

    def forward(self, raw_data, require = False):
        # Split data
        data = copy.deepcopy(raw_data)
        words = [sentence['words'] for sentence in data]
        heads = [sentence['heads'] for sentence in data]
        labels = [sentence['labels'] for sentence in data]
        if self.use_grl:
            domains = [sentence['domain'] for sentence in data]
        # Get length after padding
        n_sentences = len(data)
        max_word_len = max([len(sentence) for sentence in words])
        max_subword_len  = 0
        for word in words:
            for subword in word:
                max_subword_len = max(max_subword_len, len(subword))
        from torch.nn.functional import pad
        # Padding
        word_mask_paddings = []
        for i in range(len(words)):
            word_mask_paddings.append([False] + [True] * (len(words[i])-1) + [False] * (max_word_len - len(words[i])))
            if len(words[i]) < max_word_len:
                words[i] += [[1]] * (max_word_len - len(words[i]))
            for j in range(max_word_len):
                if len(words[i][j]) < max_subword_len:
                    words[i][j] += [1] * (max_subword_len - len(words[i][j]))
        word_paddings = torch.tensor(words).to(self.device)
        head_list = []
        for head in heads:
            head_list += head
        label_list = []
        for label in labels:
            label_list += label
        if self.use_grl:
            domain_paddings = torch.tensor([[domain] * max_word_len for domain in domains]).to(self.device)

        # Getting contexual embedding
        if self.embedding_type in ['bert', 'roberta', 'mamba']:
            embedding_output = self.encoder(word_paddings)
            '''
            embedding_output = self.encoder(word_paddings, attention_mask=attention_mask).last_hidden_state
            new_embedding_output = torch.stack([torch.cat((embedding[padding],
                                                           torch.zeros(max_word_len - len(embedding[padding]),
                                                                       self.encoder_config.hidden_size))) for
                                                embedding, padding in zip(embedding_output, word_mask_paddings)])
            embedding_output = new_embedding_output
            '''
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
        word_mask_paddings = torch.flatten(torch.tensor(word_mask_paddings).to(self.device))
        unmasked_arc_scores = torch.flatten(arc_scores, 0, 1)
        unmasked_label_scores = torch.flatten(label_scores, 0, 1)

        unmasked_arc_scores = unmasked_arc_scores[word_mask_paddings]
        unmasked_label_scores = unmasked_label_scores[word_mask_paddings]
        head_list = torch.tensor(head_list).to(self.device)
        label_list = torch.tensor(label_list).to(self.device)

        unmasked_label_scores = unmasked_label_scores[
            [torch.arange(len(head_list)), head_list]]

        if self.use_grl:
            unmasked_domain_scores = torch.flatten(domain_scores, 0, 1)[word_mask_paddings]
            unmasked_domain_paddings = torch.flatten(domain_paddings, 0, 1)[word_mask_paddings]
        # Calculate loss
        
        arc_loss = self.loss_fn(unmasked_arc_scores, head_list)
        label_loss = self.loss_fn(unmasked_label_scores, label_list)
        loss = arc_loss + label_loss

        if self.use_grl:
            grl_loss = self.loss_fn(unmasked_domain_scores, unmasked_domain_paddings)
            loss = (1.0 - self.grl_loss_rate) * loss + self.grl_loss_rate * grl_loss

        # Get predicted heads & labels
        predicted_heads = unmasked_arc_scores.argmax(-1)
        predicted_labels = unmasked_label_scores.argmax(-1)

        # Counting correct heads & labels
        correct_heads = int((predicted_heads == head_list).sum())
        correct_labels = int((predicted_labels == label_list).sum())
        n_words = word_mask_paddings.sum()

        return loss

    def evaluate(self, raw_data):
        # Turn off gradient
        with torch.no_grad():
            data = copy.deepcopy(raw_data)
            words = [sentence['words'] for sentence in data]
            heads = [sentence['heads'] for sentence in data]
            labels = [sentence['labels'] for sentence in data]

            # Get length after padding
            n_sentences = len(data)
            max_word_len = max([len(sentence) for sentence in words])
            max_subword_len  = 0
            for word in words:
                for subword in word:
                    max_subword_len = max(max_subword_len, len(subword))
            from torch.nn.functional import pad
            # Padding
            word_mask_paddings = []
            for i in range(len(words)):
                word_mask_paddings.append([False] + [True] * (len(words[i])-1) + [False] * (max_word_len - len(words[i])))
                if len(words[i]) < max_word_len:
                    words[i] += [[1]] * (max_word_len - len(words[i]))
                for j in range(max_word_len):
                    if len(words[i][j]) < max_subword_len:
                        words[i][j] += [1] * (max_subword_len - len(words[i][j]))
            word_paddings = torch.tensor(words)
            head_list = []
            for head in heads:
                head_list += head
            label_list = []
            for label in labels:
                label_list += label
    
            # Getting contexual embedding
            if self.embedding_type in ['bert', 'roberta', 'mamba']:
                embedding_output = self.encoder(word_paddings)
                '''
                embedding_output = self.encoder(word_paddings, attention_mask=attention_mask).last_hidden_state
                new_embedding_output = torch.stack([torch.cat((embedding[padding],
                                                               torch.zeros(max_word_len - len(embedding[padding]),
                                                                           self.encoder_config.hidden_size))) for
                                                    embedding, padding in zip(embedding_output, word_mask_paddings)])
                embedding_output = new_embedding_output
                '''
            # Send the embedding into MLPs
            arc_head = self.head_mlp_arc(embedding_output)
            arc_dep = self.dep_mlp_arc(embedding_output)
            label_head = self.head_mlp_label(embedding_output)
            label_dep = self.dep_mlp_label(embedding_output)
            # Biaffine attention
            arc_scores = self.biaffine_arc(arc_dep, arc_head)
            label_scores = self.biaffine_label(label_dep, label_head).permute(0, 2, 3, 1)
            # Unmask
            word_mask_paddings = torch.flatten(torch.tensor(word_mask_paddings).to(self.device))
            unmasked_arc_scores = torch.flatten(arc_scores, 0, 1)[word_mask_paddings]
            unmasked_label_scores = torch.flatten(label_scores, 0, 1)[word_mask_paddings]
    
            head_list = torch.tensor(head_list).to(self.device)
            label_list = torch.tensor(label_list).to(self.device)
    
            unmasked_label_scores = unmasked_label_scores[
                [torch.arange(len(head_list)), head_list]]
    
            # Get predicted heads & labels
            predicted_heads = unmasked_arc_scores.argmax(-1)
            predicted_labels = unmasked_label_scores.argmax(-1)
    
            # Counting correct heads & labels
            n_correct_heads = int((predicted_heads == head_list).sum())
            n_correct_labels = int(((predicted_labels == label_list) & (predicted_heads == head_list)).sum())
            n_words = int(word_mask_paddings.sum())
            # print('uas:', correct_heads/n_words*100)
            # print('las:', correct_labels/n_words*100)
        return n_words, n_correct_heads, n_correct_labels, predicted_heads, predicted_labels

    def Train(self, n_epochs, logger):
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
                self.scheduler.step()
            avg_loss = stats['train_loss'] / n_batches
            logger.info(f'Epoch {epoch_id + 1}: {avg_loss}, {datetime.datetime.now() - start_time} seconds.')
            current_lr = sum(self.scheduler.get_last_lr()) / len(self.scheduler.get_last_lr())
            logger.info(f'lr: {current_lr}')
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
                torch.save(self.state_dict(), self.model_save_dir)
        with open(self.config_save_dir, 'w') as out_file:
            config_data = {'pos_tag_vocab': self.pos_tag_vocab, 'label_vocab': self.label_vocab,
                           'domain_vocab': self.domain_vocab}
            json.dump(config_data, out_file)
        logger.info(f'Best record on epoch {best_epoch}:')
        logger.info(f'Dev  set:\tUAS: {round(best_dev_uas, 2)}\tLAS: {round(best_dev_las, 2)}')
        logger.info(f'Test set:\tUAS: {round(best_test_uas, 2)}\tLAS: {round(best_test_las, 2)}')

    def Eval(self, dataset, require_preprocessing=False, logger=None, mode='evaluate'):
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
        predicted_heads_list = []
        predicted_labels_list = []
        for batch in range(0, len(eval_data), self.batch_size):
            start_time = datetime.datetime.now()
            data = eval_data[batch:min(batch + self.batch_size, len(eval_data))]
            n_words, n_correct_heads, n_correct_labels, predicted_heads, predicted_labels = self.evaluate(data)
            records['words'] += n_words
            records['correct_heads'] += n_correct_heads
            records['correct_labels'] += n_correct_labels
            predicted_heads_list += predicted_heads
            predicted_labels_list += predicted_labels
        uas = records['correct_heads'] / records['words'] * 100
        las = records['correct_labels'] / records['words'] * 100
        if logger is not None:
            logger.info(f'UAS: {round(uas, 2)}')
            logger.info(f'LAS: {round(las, 2)}')
        if mode == 'evaluate':
            return uas, las
        if mode == 'predict':
            return predicted_heads_list, predicted_labels_list
