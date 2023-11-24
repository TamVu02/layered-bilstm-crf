import os
import sys
os.environ['CHAINER_SEED'] = '0'
import random
import numpy as np
np.random.seed(0)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle

import chainer.functions as F
from chainer import iterators
from chainer import cuda
from chainer import serializers

from src.model.layered_model import Model, Evaluator, Updater
from src.model.loader import load_sentences, update_tag_scheme, parse_config
from src.model.loader import prepare_dataset
from src.model.utils import evaluate

seed = 11
random.seed(seed)


def predict(data_iter, model, mode):
    """
    Iterate data with well - trained model
    """
    for batch in data_iter:
        raw_words = [x['str_words'] for x in batch]
        words = [model.xp.array(x['words']).astype('i') for x in batch]
        chars = [model.xp.array(y).astype('i') for x in batch for y in x['chars']]
        tags = model.xp.vstack([model.xp.array(x['tags']).astype('i') for x in batch])

        # Init index to keep track of words
        index_start = model.xp.arange(F.hstack(words).shape[0])
        index_end = index_start + 1
        index = model.xp.column_stack((index_start, index_end))

        # Maximum number of hidden layers = maximum nested level + 1
        max_depth = len(batch[0]['tags'][0])
        sentence_len = np.array([x.shape[0] for x in words])
        section = np.cumsum(sentence_len[:-1])
        predicts_depths = model.xp.empty((0, int(model.xp.sum(sentence_len)))).astype('i')

        for depth in range(max_depth):
            next, index, extend_predicts, words, chars = model.predict(chars, words, tags[:, depth], index, mode)
            predicts_depths = model.xp.vstack((predicts_depths, extend_predicts))
            if not next:
                break

        predicts_depths = model.xp.split(predicts_depths, section, axis=1)
        ts_depths = model.xp.split(model.xp.transpose(tags), section, axis=1)
        yield ts_depths, predicts_depths, raw_words


def load_mappings(mappings_path):
    """
    Load mappings of:
      + id_to_word
      + id_to_tag
      + id_to_char
    """
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
        id_to_word = mappings['id_to_word']
        id_to_char = mappings['id_to_char']
        id_to_tag = mappings['id_to_tag']

    return id_to_word, id_to_char, id_to_tag


def main(config_path):
    args = parse_config(config_path)

    # Load sentences
    test_sentences = load_sentences(args["path_dev"], args["replace_digit"])
    # random.shuffle(all_sentences)
    # total_sample = len(all_sentences)
    # ratio = (int)(total_sample*90/100)
    # test_sentences=all_sentences[ratio:]

    '''
    test_sentences=[[['Trong','O','O','O','O'],
    ['cuộc','B-BATTLE','O','O','O'],
    ['khởi','I-BATTLE','O','O','O'],
    ['nghĩa','I-BATTLE','O','O','O'],
    ['của','I-BATTLE','O','O','O'],
    ['quân','I-BATTLE','B-MIL','O','O'],
    ['Nguyễn,','I-BATTLE','I-MIL','B-ORG','O'],
    ['nhà','B-ORG','O','O','O'],
    ['Nguyễn','I-ORG','O','O','O'],
    ['chống','O','O','O','O'],
    ['trả','O','O','O','O'],
    ['rất','O','O','O','O'],
    ['quyết','O','O','O','O'],
    ['liệt.','O','O','O','O'],
    ['Vua','O','O','O','O'],
    ['Nguyễn','B-PER','O','O','O'],
    ['Thái','I-PER','O','O','O'],
    ['Tông','I-PER','O','O','O'],
    ['đã','O','O','O','O'],
    ['quyết','O','O','O','O'],
    ['định','O','O','O','O'],
    ['đóng','O','O','O','O'],
    ['đô','O','O','O','O'],
    ['ở','O','O','O','O'],
    ['Hoa','B-LOC','O','O','O'],
    ['Lư','I-LOC','O','O','O']]]
    '''
    def covert_text_to_iob_format(text):
        iob_data=list()
        for word in text.split():
            cur_iob=list()
            cur_iob.append(word)
            cur_iob+= 'O' * 4
            iob_data.append(cur_iob)
        return [iob_data]
    #text='Trong khoảng 00 thế kỉ đầu sau Công nguyên, hàng loạt quốc gia nhỏ đã được hình thành và phát triển ở khu vực phía nam Đông Nam Á như Vương quốc Cham-pa ở vùng Trung Bộ Việt Nam, Vương quốc Phù Nam ở hạ lưu sông Mê Công, các vương quốc ở hạ lưu sông Mê Nam và trên các đảo của In-đô-nê-xi-a. Thời ấy, các quốc gia này còn nhỏ bé, phân tán trên các địa bàn hẹp, sống riêng rẻ và nhiều khi tranh chấp lẫn nhau. Đó cũng chính là nguyên nhân dẫn tới sự sụp đổ của các vương quốc cổ, để rồi, trên cơ sở đó hình thành nên các quốc gia phong kiến dân tộc hùng mạnh sau này.'
    #text=input('Input for predict nested ner: ')
    #test_sentences=covert_text_to_iob_format(text)

    # Update tagging scheme (IOB/IOBES)
    update_tag_scheme(test_sentences, args["tag_scheme"])

    # Load mappings from disk
    id_to_word, id_to_char, id_to_tag = load_mappings(args["mappings_path"])
    word_to_id = {v: k for k, v in id_to_word.items()}
    char_to_id = {v: k for k, v in id_to_char.items()}
    tag_to_id  = {v: k for k, v in id_to_tag.items()}

    # Index data
    test_data = prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id, None, args["lowercase"])
    test_iter = iterators.SerialIterator(test_data, args["batch_size"], repeat=False, shuffle=False)

    model = Model(len(word_to_id), len(char_to_id), len(tag_to_id), args)

    serializers.load_npz(args['path_model'], model)
    serializers.save_npz('nested_ner.npz', model)

    model.id_to_tag = id_to_tag
    model.parameters = args

    device = args['gpus']
    if device['main'] >= 0:
        cuda.get_device_from_id(device['main']).use()
        model.to_gpu()

    pred_tags = []
    gold_tags = []
    words = []

    # Collect predictions
    for ts, ys, xs in predict(test_iter, model, args['mode']):
        gold_tags.extend(ts)
        pred_tags.extend(ys)
        words.extend(xs)

    max_tag_pred=len(pred_tags[0])
    for sample in range (len(pred_tags)):
        max_tag_pred=len(pred_tags[sample])
        if(max_tag_pred<=2):
            continue
        for i in range(len(words[sample])):
            print(words[sample][i],'\t',end=' ')
            for j in range (max_tag_pred):
                tag=pred_tags[sample][j][i].item()
                print(id_to_tag.get(tag),end='\t')
            print('\n')
        print('\n')

    evaluate(model, pred_tags, gold_tags, words)


if __name__ == '__main__':
    main('/content/layered_bilstm_crf/src/config_test')
