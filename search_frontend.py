import math
import pickle
import csv
from flask import Flask, request, jsonify
import inverted_index_colab
import inverted_index_gcp
import requests

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
from contextlib import closing

class MyFlaskApp(Flask):
    def read_posting_list(self, w):
        inverted= self.inverted
        with closing(inverted_index_gcp.MultiFileReader()) as reader:
            locs = inverted.posting_locs[w]
            s = str(locs[0][0])
            locs = [("C:\\Users\\HP\\Desktop\\project data\\postings_gcp\\"+s, locs[0][1])]
            b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list

    def read_posting_list_title(self, w):
        inverted = self.inverted_title
        with closing(inverted_index_gcp.MultiFileReader()) as reader:
            locs = inverted.posting_locs[w]
            s = str(locs[0][0])
            locs = [("C:\\Users\\HP\\Desktop\\project data\\postings_gcp_title\\"+s, locs[0][1])]
            b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list

    def read_posting_list_anchor(self, w):
        inverted = self.inverted_anchor
        with closing(inverted_index_gcp.MultiFileReader()) as reader:
            locs = inverted.posting_locs[w]
            s = str(locs[0][0])
            locs = [("C:\\Users\\HP\\Desktop\\project data\\postings_gcp_anchor\\"+s, locs[0][1])]
            b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list
    # def read_posting_list2(self, w):
    #     inverted = self.inverted2
    #     with closing(inverted_index_gcp.MultiFileReader()) as reader:
    #         locs = inverted.posting_locs[w]
    #         s = str(locs[0][0])
    #         locs=[("C:\\Users\\HP\\Downloads\\postings_gcp_0_000 (1).bin")]
    #         b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
    #         posting_list = []
    #         for i in range(inverted.df[w]):
    #             doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
    #             tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
    #             posting_list.append((doc_id, tf))
    #         return posting_list

    def search(self, query):
        body_top = app.get_top_pages_by_body(query)
        # print(body_top)
        title_top = app.get_top_pages_by_title(query)
        # print(title_top)

        merge = {}
        for i in body_top.keys():
            body_top[i] = body_top[i] * 0.6
            merge[i] = body_top[i]
        for j in title_top.keys():
            title_top[j] = title_top[j] * 1000
            if j in merge:
                merge[j] = merge[j] + title_top[j]
            else:
                merge[j] = title_top[j]
        top = dict(sorted(merge.items(), key=lambda item: item[1], reverse=True))
        #print(top)
        id = app.get_top_100_id(top)
        id_title = app.get_id_title(id)
        return id_title

    def get_top_pages_by_body(self, query):
        tfidf = {}
        for word in query:
            if word not in self.inverted.df.keys():
                continue
            word_list = self.read_posting_list(word)
            #word_list = [(12,2), (25,3), (39,2)]
            # doc_len = {12:4, 14:3, 173:10}
            # self.corpus_size = 6,300,000

            # calculate idf
            df = len(word_list)  # 3
            idf = math.log(self.corpus_size / df, 2)  # log2(6M/3)

            for id_tf in word_list:
                doc_id = id_tf[0]  # 12
                tf = id_tf[1]  # 2
                if tf==0:
                    continue
                doc_len_normal = self.id_len_dict[doc_id]  # 4, 3, 10
                if doc_len_normal < 100:
                    doc_len_normal *= 10000
                elif doc_len_normal > 1500:
                    doc_len_normal /= 100
                elif doc_len_normal > 500:
                    doc_len_normal /= 10
                elif doc_len_normal > 300:
                    doc_len_normal /= 10
                elif doc_len_normal > 50:
                    doc_len_normal /= 5

                # elif doc_len_normal > 700:
                #     doc_len_normal /= 1000

                normalized_tf = tf / doc_len_normal
                tfidf_value = normalized_tf * idf
                if doc_id in tfidf.keys():
                    tfidf[doc_id] = tfidf[doc_id] + tfidf_value
                else:
                    tfidf[doc_id] = tfidf_value

            # tfidf = {12:0.5, 14:1, 173:0.2}
            # tfidf = {12:(0.5 * 0.41),14:(1 * 0.41), 173:(0.2 * 0.41)}

        tfidf = dict(sorted(tfidf.items(), key=lambda item: item[1], reverse=True))
        # tfidf = {14:0.41, 12:0.205, 173:0.08}

        return tfidf
        # best_100_id = list(tfidf.keys())[0:100]
        # return best_100_id

    def get_top_pages_by_anchor(self, query):
        bool_dict = {}
        doc_id = 0
        tf = 0
        for word in query:
            if word not in self.inverted_anchor.df.keys():
                continue
            word_list = self.read_posting_list_anchor(word)
            for id_tf in word_list:
                doc_id = id_tf[0]  # 12
                tf = id_tf[1]  # 1
                if doc_id in bool_dict.keys():
                    bool_dict[doc_id] = bool_dict[doc_id] + tf
                else:
                    bool_dict[doc_id] = tf
        bool_dict = dict(sorted(bool_dict.items(), key=lambda item: item[1], reverse=True))
        return bool_dict

    def get_top_pages_by_title(self, query):
        bool_dict = {}
        doc_id = 0
        tf = 0
        for word in query:
            if word not in self.inverted_title.df.keys():
                continue
            word_list = self.read_posting_list_title(word)
            for id_tf in word_list:
                doc_id = id_tf[0]  # 12
                tf = id_tf[1]  # 1
                if doc_id in bool_dict.keys():
                    bool_dict[doc_id] = bool_dict[doc_id] + tf
                else:
                    bool_dict[doc_id] = tf
        bool_dict = dict(sorted(bool_dict.items(), key=lambda item: item[1], reverse=True))
        return bool_dict


        # best_100_id = list(bool_dict.keys())[0:100]
        # return best_100_id

    def get_id_title(self, id_list):
        titles = []
        for i in range(len(id_list)):
            titles.append(self.id_title_dict[id_list[i]])
        id_title = list(zip(id_list, titles))
        return id_title

    def run(self, host=None, port=None, debug=None, **options):

        self.inverted = inverted_index_gcp.InvertedIndex()
        self.inverted_title = inverted_index_gcp.InvertedIndex()
        self.inverted_anchor = inverted_index_gcp.InvertedIndex()
        self.corpus_size = 6348910
        self.id_len_dict = None
        self.id_title_dict = None
        self.id_page_rank_dict = {}
        self.id_page_rank_dict2 = {}
        self.id_page_view_dict = {}
        self.id_page_view_dict2 = {}

        # #load BODY index
        # with open('C:\\Users\\HP\\Desktop\\project data\\postings_gcp\\index.pkl', 'rb') as f:
        #     data = pickle.load(f)
        #     self.inverted.df = data.df
        #     self.inverted.posting_locs = data.posting_locs
        #
        # # load TITLE index
        # with open('C:\\Users\\HP\\Desktop\\project data\\postings_gcp_title\\index.pkl', 'rb') as f:
        #     data = pickle.load(f)
        #     self.inverted_title.df = data.df
        #     self.inverted_title.posting_locs = data.posting_locs
        # for i in range(0, 124):
        #     path = 'C:\\Users\\HP\\Desktop\\project data\\postings_gcp_title\\' + str(i) + '_posting_locs.pickle'
        #     with open(path, 'rb') as f:
        #         data = pickle.load(f)
        #         self.inverted_title.posting_locs.update(data)

        # # load ANCHOR TEXT index
        # with open('C:\\Users\\HP\\Desktop\\project data\\postings_gcp_anchor\\index.pkl', 'rb') as f:
        #     data = pickle.load(f)
        #     self.inverted_anchor.df = data.df
        #     self.inverted_anchor.posting_locs = data.posting_locs
        # for i in range(0, 124):
        #     path = 'C:\\Users\\HP\\Desktop\\project data\\postings_gcp_anchor\\' + str(i) + '_posting_locs.pickle'
        #     with open(path, 'rb') as f:
        #         data = pickle.load(f)
        #         self.inverted_anchor.posting_locs.update(data)


        # load {id: page_view} dict
        #with open('page_views.pkl', 'rb') as f:
        with open('C:\\Users\\HP\\Desktop\\project data\\page_views.pkl', 'rb') as f:
            data = pickle.load(f)
            self.id_page_view_dict = data

        # # load {id: len} dict
        # with open('C:\\Users\\HP\\Desktop\\project data\\docs_total_tokens.pkl', 'rb') as f:
        #     self.id_len_dict = pickle.load(f)
        #
        # # load {id: title} dict
        # with open('C:\\Users\\HP\\Desktop\\project data\\id_title_dict.pkl', 'rb') as f:
        #     self.id_title_dict = pickle.load(f)
        #     print()

        # load {id: page_rank} dict
        with open('C:\\Users\\HP\\Desktop\\project data\\id_page_rank.xls.csv', mode='r') as inp:
            reader = csv.reader(inp)
            i = 0
            for row in reader:
                i+=1
                if i > 4000000:
                    self.id_page_rank_dict2[row[0]]=row[1]
                else:
                    self.id_page_rank_dict[row[0]]=row[1]
        #import json
        print()

        # Opening JSON file
        # f = open('queries_train.json')
        #
        # # returns JSON object as
        # # a dictionary
        # queries = json.load(f)
        # print(queries)
        #
        # eval_body = {}
        # eval_title= {}
        # eval_search={}
        # for i in queries.keys():
        #     key = i
        #     true = queries[i]
        #     if ' ' in i:
        #         i = i.split(' ')
        #     else:
        #         i =[i]
        #     #pred_id_score = app.get_top_pages_by_title(i)
        #     # pred_id_score = app.get_top_pages_by_body(i)
        #     pred_id_score = app.search(i)
        #     print("pred: ", pred_id_score)
        #     ids=[]
        #     for i in range(len(pred_id_score)):
        #         add = pred_id_score[i][0]
        #         # print("add", add)
        #         # print("ids",ids)
        #         ids.append(pred_id_score[i][0])
        #     #pred_id = app.get_top_100_id(ids)
        #     avg = app.average_precision(true, ids)
        #     eval_search[key] = avg
        # print(eval_search)
        # body_values = eval_search.values()
        # with open('eval_search.csv', 'w') as f:
        #     write = csv.writer(f)
        #     write.writerow(body_values)
        # print()
            # *** evaluation ***
        # for i in x.keys():
        #     q = i.split(' ')
        #     true = x[i]
        #     print(i)
        #     print(true)
        #     pred = app.get_top_pages_by_title(q)
        #     id = app.get_top_100_id(pred)
        #     x = app.average_precision(true,id)
        #     print(x)

        # a= self.read_posting_list2('perry')
        #a = app.get_page_rank_by_id([1,2,3])
        # a = app.get_top_pages_by_title(["python"])
        # a = app.get_top_100_id(a)
        # b = [23862, 23329, 53672527, 21356332, 4920126, 5250192, 819149, 46448252, 83036, 88595, 18942, 696712, 2032271, 1984246, 5204237, 645111, 18384111, 3673376, 25061839, 271890, 226402, 2380213, 1179348, 15586616, 50278739, 19701, 3596573, 4225907, 19160, 1235986, 6908561, 3594951, 18805500, 5087621, 25049240, 2432299, 381782, 9603954, 390263, 317752, 38007831, 2564605, 13370873, 2403126, 17402165, 23678545, 7837468, 23954341, 11505904, 196698, 34292335, 52042, 2247376, 15858, 11322015, 13062829, 38833779, 7800160, 24193668, 440018, 54351136, 28887886, 19620, 23045823, 43003632, 746577, 1211612, 8305253, 14985517, 30796675, 51800, 964717, 6146589, 13024, 11583987, 57294217, 27471338, 5479462]


        # print(x)

        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

    def get_page_rank_by_id(self, wiki_ids):
        res = []
        #wiki_ids = [14673744, 24899468]
        for i in wiki_ids:
            i = str(i)
            if i in self.id_page_rank_dict:
                res.append(self.id_page_rank_dict[i])
            elif i in self.id_page_rank_dict2:
                res.append(self.id_page_rank_dict2[i])
        print(res)
        return res

    def get_page_view(self, wiki_ids):
        res = []
        #wiki_ids = [14673744, 24899468]
        for i in wiki_ids:
            # i = str(i)
            if i in self.id_page_view_dict:
                res.append(self.id_page_view_dict[i])
            elif i in self.id_page_view_dict2:
                res.append(self.id_page_view_dict2[i])
        print(res)
        return res

    def get_top_100_id(self, dict_score):
        best_100_id = list(dict_score.keys())[0:100]
        return best_100_id

    def average_precision(self, true_list, predicted_list, k=40):
        true_set = frozenset(true_list)
        predicted_list = predicted_list[:k]
        precisions = []
        for i, doc_id in enumerate(predicted_list):
            if doc_id in true_set:
                prec = (len(precisions) + 1) / (i + 1)
                precisions.append(prec)
        if len(precisions) == 0:
            return 0.0
        return round(sum(precisions) / len(precisions), 3)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    query = query.split()
    for i in range(len(query)):
        query[i] = query[i].lower()
    print(query)
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = app.search(query)
    print(res)
    # top_100_id = app.get_top_100_pages(query)
    # top_100_id_title = app.get_id_title(top_100_id)
    # res = top_100_id_title
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    query = query.split()
    for i in range(len(query)):
        query[i] = query[i].lower()
    # print(query)
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    top_id_score = app.get_top_pages_by_body(query)
    top_100_id = app.get_top_100_id(top_id_score)
    top_100_id_title = app.get_id_title(top_100_id)
    res = top_100_id_title
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    query = query.split()
    for i in range(len(query)):
        query[i] = query[i].lower()
    if len(query) == 0:
      return jsonify(res)
    print(query)
    # BEGIN SOLUTION
    top_id_score_title = app.get_top_pages_by_title(query)
    id = app.get_top_100_id(top_id_score_title)
    id_title = app.get_id_title(id)
    res = id_title
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    query = query.split()
    for i in range(len(query)):
        query[i] = query[i].lower()
    if len(query) == 0:
        return jsonify(res)
    print(query)
    # BEGIN SOLUTION
    top_id_score_anchor = app.get_top_pages_by_anchor(query)
    id = app.get_top_100_id(top_id_score_anchor)
    id_title = app.get_id_title(id)
    res = id_title
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = app.get_page_rank_by_id(wiki_ids)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = app.get_page_view(wiki_ids)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
    import requests
    x = requests.post(url="http://192.168.14.2:8080/get_pagerank", json=[1, 2, 3])
    print(x.json())
    #request.post(url="http://0.0.0.0:8080/get_pagerank", json=[1, 2, 3])



    x = requests.post(url="http://192.168.14.2:8080/get_pagerank", json=[1, 2, 3])
    print(x.json())