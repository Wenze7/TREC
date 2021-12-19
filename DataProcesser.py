import pandas as pd
import matplotlib.pyplot as plt
import pickle


def remove_stop_words(texts):
    stop_words = set(['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'])
    for i in range(len(texts)):
        text = texts[i]
        text_list = text.split(' ')
        text_list = [item for item in text_list if item not in stop_words]
        text = ' '.join(text_list)
        text = text.lower()
        texts[i] = text

    return texts


def read_id2texts(path):
    data = pd.read_csv(path, sep='\t', header=None)
    data_ids = data[0].tolist()
    data_texts = data[1].tolist()
    data_texts = remove_stop_words(data_texts)
    vision(data_texts)
    id2texts = {int(data_ids[i]): data_texts[i] for i in range(len(data))}
    return id2texts


def read_triples(path):
    data = pd.read_csv(path, sep='\t', header=None)
    query_ids = data[0].tolist()
    pos_passage_ids = data[1].tolist()
    neg_passage_ids = data[2].tolist()
    triples = [(int(query_ids[i]), int(pos_passage_ids[i]), int(neg_passage_ids[i])) for i in range(len(data))]
    return triples


def read_top_file_data(path):
    data = pd.read_csv(path, sep='\t', header=None)
    query_ids = data[0].tolist()
    passage_ids = data[1].tolist()
    query_texts = data[2].tolist()
    passage_texts = data[3].tolist()
    query_texts = remove_stop_words(query_texts)
    passage_texts = remove_stop_words(passage_texts)
    id2queries = {int(query_ids[i]): query_texts[i] for i in range(len(data))}
    id2passages = {int(passage_ids[i]): passage_texts[i] for i in range(len(data))}
    tuples = [(int(query_ids[i]), int(passage_ids[i])) for i in range(len(data))]

    return id2passages, id2queries, tuples


def read_data(args):
    id2queries = read_id2texts(args.train_queries_path)
    id2passages = read_id2texts(args.train_passages_path)
    triples = read_triples(args.train_triples_path)
    id2passages_valid, id2queries_valid, valid_tuples = read_top_file_data(args.valid_top_file_path)
    id2passages_test, id2queries_test, valid_test = read_top_file_data(args.test_top_file_path)

    train_data = [id2queries, id2passages, triples]
    valid_data = [id2passages_valid, id2queries_valid, valid_tuples]
    test_data = [id2passages_test, id2queries_test, valid_test]

    pickle.dump(train_data, open(args.save_path + '/train_data.pkl', 'wb'))
    pickle.dump(valid_data, open(args.save_path + '/valid_data.pkl', 'wb'))
    pickle.dump(test_data, open(args.save_path + '/test.pkl', 'wb'))



    # k1 = list(id2passages.keys())
    # k2 = list(id2passages_test.keys())
    # k3 = list(id2passages_valid.keys())
    # print(len(k1), len(k2), len(k3))
    # sk1 = set(k1)
    # sk2 = set(k2)
    # sk3 = set(k3)
    #
    # print(len(sk1), len(sk2), len(sk3))
    #
    # for k in (sk1 & sk2 & sk3):
    #     print(id2passages[k])
    #     print(id2passages_valid[k])
    #     print(id2passages_test[k])


def vision(L):
    cnt = {}
    for li in L:
        l = len(li)
        if l not in cnt:
            cnt[l] = 0
        cnt[l] += 1
    cnt = sorted(cnt.items(), key=lambda d: d[1])
    x = [k[0] for k in cnt]
    y = [k[1] for k in cnt]

    plt.bar(x, y)
    plt.show()


