import conllu
import pickle


def load_data(folder):
    """Loads raw data and hanzi (chinese characters) if it is already downloaded.
    To download, refer to get_data.py.
    :param
    folder (str): location where to load files

    :return:
    chinese_train (dict): raw training data in Conllu format
    chinese_val (dict): raw validation data in Conllu format
    chinese_test (dict): raw test data in Conllu format
    hanzi (list): Hanzi lookup dictionary

    """

    with open(f"{folder}/zh_gsd-ud-train.conllu", "r", encoding="utf-8") as train:
        chinese_train = conllu.parse(train.read())
    train.close()

    with open(f"{folder}/zh_gsd-ud-dev.conllu", "r", encoding="utf-8") as val:
        chinese_val = conllu.parse(val.read())
    val.close()

    with open(f"{folder}/zh_gsd-ud-test.conllu", "r", encoding="utf-8") as test:
        chinese_test = conllu.parse(test.read())
    test.close()

    with open('data/hanzi.pkl', 'rb') as hanzi_file:
        hanzi = pickle.load(hanzi_file)
    hanzi_file.close()

    return chinese_train, chinese_val, chinese_test, hanzi


def get_pos_map(data):
    """Maps POS to numerical labels

    :param
    data (list): raw data in conllu format

    :return:
    tags_map (dictionary): {pos_tag: index}.

    """

    train_pos = [dic['upostag'] for dic in sum(data, [])]
    text_labels = list(set(train_pos))
    tags_map = dict()

    for idx in range(len(text_labels)):
        tags_map[text_labels[idx]] = idx

    return tags_map


def categorical_accuracy(predictions, tags):
    """Returns accuracy per sentence: 7/10 right = 0.7

    :param predictions: model output
    :param tags: actual labels

    :return: (int): per-sentence accuracy

    """

    max_preds = predictions.argmax(dim=1, keepdim=True)  # max probability index
    correct = max_preds.squeeze(1).eq(tags)
    return correct.sum().item() / tags.shape[0]
