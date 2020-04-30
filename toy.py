import torch
import numpy as np
import torch.optim as optim

from bilstm_crf import BiLSTM_CRF
from const_num import START_TAG, STOP_TAG, EMBEDDING_DIM, HIDDEN_DIM
from evluate import Acc
from helper import prepare_sequence
from progressbar import ProgressBar

torch.manual_seed(1)


def get_train_data():
    # Make up some training data
    training_data = [(
        "上 海 大 桥 相 当 于 苏 州 大 小".split(),
        "B I I I O O O B I O O".split()
    ), (
        "苏 州 不 如 无 锡 大".split(),
        "B I O O B I O".split()
    )]
    return training_data


def get_word_to_ix(training_data, test_data=None):
    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    if test_data is not None:
        for sentence, tags in test_data:
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
    return word_to_ix


def train(model, training_data, epoch_num=300):
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(
            epoch_num):  # again, normally you would NOT do 300 epochs, it is toy data
        pbar = ProgressBar(n_total=len(training_data), desc='Training')
        step = 0
        acc = Acc()
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)
            predict=model(sentence_in)
            predict_tags = np.array(predict[1])
            predict_tags = torch.from_numpy(predict_tags)
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
            acc.update(predict_tags, targets)
            pbar(step, {'loss': loss.item(),"acc":acc.acc})
            step += 1


def test(model, test_data):
    # Check predictions after training
    acc=Acc()
    with torch.no_grad():
        for test_sentence,test_tag in test_data:
            precheck_sent = prepare_sequence(test_sentence, word_to_ix)
            predict = model(precheck_sent)
            predict_tags = np.array(predict[1])
            predict_tags = torch.from_numpy(predict_tags)
            precheck_tags = torch.tensor([tag_to_ix[t] for t in test_tag], dtype=torch.long)
            acc.update(predict_tags,precheck_tags)
        print("acc:{%.4f}" % acc.acc)

# We got it!
if __name__ == '__main__':
    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
    training_data = get_train_data()
    word_to_ix = get_word_to_ix(training_data=training_data)
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    test_data = training_data
    test(model, test_data)
    train(model, training_data)
    test(model, test_data)
