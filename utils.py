def Get_subwords_mask_BERT(sentence, subword_list):
    # Get the mask to convert the output of BERT Embedding from subword-level to word-level.
    # For each word, we will keep the LAST subword and remove others.
    padding = []
    pos = 0
    word = ''
    for subword in subword_list:
        if len(subword) > 2 and subword[:2] == '##':
            subword = subword[2:]
        word += subword
        if word == sentence[pos]:
            padding.append(True)
            pos += 1
            word = ''
        else:
            padding.append(False)
    assert (pos == len(sentence))
    assert (word == '')
    padding = padding
    return padding


def Get_subwords_mask_RoBERTa(sentence, subword_list):
    # Get the mask to convert the output of BERT Embedding from subword-level to word-level.
    # For each word, we will keep the LAST subword and remove others.
    padding = []
    pos = 0
    word = ''
    for subword in subword_list:
        subword = subword.replace('Ġ', '').replace('▁', '')
        word += subword
        if word == sentence[pos]:
            padding.append(True)
            pos += 1
            word = ''
        else:
            padding.append(False)
    assert (pos == len(sentence))
    assert (word == '')
    padding = padding
    return padding


def Get_subwords_mask_PhoBERT(sentence, subword_list):
    # Get the mask to convert the output of BERT Embedding from subword-level to word-level.
    # For each word, we will keep the LAST subword and remove others.
    padding = []
    pos = 0
    word = ''
    for subword in subword_list:
        if len(subword) > 2 and subword[-2:] == '@@':
            subword = subword[:-2]
        word += subword
        if word == sentence[pos]:
            padding.append(True)
            pos += 1
            word = ''
        else:
            padding.append(False)
    assert (pos == len(sentence))
    assert (word == '')
    padding = padding
    return padding
