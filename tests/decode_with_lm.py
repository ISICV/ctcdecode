import numpy as np
import ctcdecode
import torch
import sys
import unicodedata


def convert_2_string(tokens, vocab, seq_len):
    output = ''.join([vocab[x] if vocab[x] not in seperate_toks else ' %s ' % vocab[x] for x in tokens[0:seq_len]])
    form = []
    for word in output.split():
        if len(word)>5:
            form.append('u' + word[1:].replace('u', '_u'))
        else:
            form.append(word)
    formatted_uxxxx_str =  ' '.join(form)
    text = ''
    for uword in formatted_uxxxx_str.split():
        for uchar in uword.split('_'):
            cur_char = uchar[1:]
            text+=chr(int(cur_char, 16))
    return text, formatted_uxxxx_str
            

def convert_2_vec(text, vocab):
    # for a batch size of 1
    vec = np.ones((len(text), len(vocab)))
    prev_char = None
    idx = 0
    for char in text:
        if prev_char == char:
            # insert ctc blank before next line
            vec = np.append(vec, np.ones((1, len(vocab))), axis=0)
            vec[idx] *= (1 - 0.7)/(len(vocab)-1)
            vec[idx][vocab.index('<ctc-blank>')] = 0.7
            idx += 1
            
        raw_hex_str = hex(ord(char))
        uxxxx_str = raw_hex_str[2:]
        if len(uxxxx_str) == 1:
            uxxxx_str = "u000%s" % uxxxx_str
        elif len(uxxxx_str) == 2:
            uxxxx_str = "u00%s" % uxxxx_str
        elif len(uxxxx_str) == 3:
            uxxxx_str = "u0%s" % uxxxx_str
        elif len(uxxxx_str) == 4:
            uxxxx_str = "u%s" % uxxxx_str
        vec[idx] *= (1 - 0.7)/(len(vocab)-1)
        vec[idx][vocab.index(uxxxx_str)] = 0.7
        prev_char = char
        idx += 1
    return vec

uxxxx_text_file = "twinkle.txt"
kenlm_ngram_model = "twinkle.ngram"
beam_size = 100
text = "twinkle, twinkle, little star,"
vocab = set()
seperate_toks = set()

with open(uxxxx_text_file, 'r') as fh:
    for line in fh:
        for uxxxx_word in line.strip().split():
            for uxxxx in uxxxx_word.split('_'):
                vocab.add(uxxxx)
                unicode_char = chr(int(uxxxx[1:], 16))
                if 'L' not in unicodedata.category(unicode_char):
                    seperate_toks.add(uxxxx)

vocab_list = ['<ctc-blank>'] + sorted(list(vocab))

probs_seq = convert_2_vec(text, vocab_list)
probs_tensor = torch.FloatTensor([probs_seq])

# CUDA build test
# probs_tensor = torch.cuda.FloatTensor([probs_seq])

print("---------------------------NO LM--------------------------------")
decoder = ctcdecode.CTCBeamDecoder(vocab_list, beam_width=beam_size, blank_id=vocab_list.index('<ctc-blank>'))
beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(probs_tensor)
for i in range(10):
    print("beam_result[{}][{}] : \ntext:{}\nuxxxx:{}".format(0, i, *convert_2_string(beam_result[0][i], vocab_list, out_seq_len[0][i])))
    print("\tscore = %f" % beam_scores[0][i])



print("\n\n---------------------------LM--------------------------------")
lm_decoder = ctcdecode.CTCBeamDecoder(vocab_list, alpha=2.0, beta=0.4, cutoff_top_n = len(vocab_list), tokenization_labels=list(seperate_toks), model_path=kenlm_ngram_model, beam_width=beam_size, blank_id=vocab_list.index('<ctc-blank>'))
beam_result_with_score, beam_scores_with_score, timesteps_with_score, out_seq_len_with_score = lm_decoder.decode(probs_tensor)
for i in range(10):
    print("beam_result[{}][{}] : \ntext:{}\nuxxxx:{}".format(0, i, *convert_2_string(beam_result_with_score[0][i], vocab_list, out_seq_len_with_score[0][i])))
    print("\tscore = %f" % beam_scores_with_score[0][i])

