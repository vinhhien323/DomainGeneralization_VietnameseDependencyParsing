
class Dataset:
    def __init__(self, directory, use_folder=False, use_domain=False):
        if use_folder:
            import glob
            file_list = glob.glob(f'{directory}/*.conllu')
        else:
            file_list = [directory]
        self.use_domain = use_domain
        self.data = []
        for file in file_list:
            self.data += self.Read(file)
        self.pos_tag_vocab = self.Create_pos_tag_vocab(self.data)
        self.label_vocab = self.Create_label_vocab(self.data)
        if self.use_domain:
            self.domain_vocab = self.Create_domain_vocab(self.data)
        else:
            self.domain_vocab = None

    def Read(self, input_file):
        file = open(input_file, 'r', encoding='utf-8')
        if self.use_domain is True:
            domain_name = input_file.split('/')[-1].replace('.conllu', '')
        data = []
        words = []
        pos_tags = []
        heads = []
        labels = []
        for line in file:
            line = line.split('\t')
            if len(line) != 10:
                if len(words) > 0:
                    sentence = dict({'words': words, 'pos_tags': pos_tags, 'heads': heads, 'labels': labels})
                    if self.use_domain is True:
                        sentence['domain'] = domain_name
                    data.append(sentence)
                    words = []
                    pos_tags = []
                    heads = []
                    labels = []
                continue
            words.append(line[1])
            pos_tags.append(line[3])
            heads.append(int(line[6]))
            labels.append(line[7])
        if len(words) > 0:
            sentence = dict({'words': words, 'pos_tags': pos_tags, 'heads': heads, 'labels': labels})
            if self.use_domain is True:
                sentence['domain'] = domain_name
            data.append(sentence)
        file.close()
        return data

    def Create_pos_tag_vocab(self, data):
        pos_tag_list = set(tag for sentence in data for tag in sentence['pos_tags'])
        pos_tag_vocab = dict({tag: id for id, tag in enumerate(pos_tag_list)})
        return pos_tag_vocab

    def Create_label_vocab(self, data):
        label_list = set(label for sentence in data for label in sentence['labels'])
        label_vocab = dict({label: id for id, label in enumerate(label_list)})
        return label_vocab

    def Create_domain_vocab(self, data):
        domain_list = set(sentence['domain'] for sentence in data)
        domain_vocab = dict({domain: id for id, domain in enumerate(domain_list)})
        return domain_vocab
