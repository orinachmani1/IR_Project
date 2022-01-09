import math
import pickle
import csv
from flask import Flask, request, jsonify
import inverted_index_colab
import inverted_index_gcp
import request

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
from contextlib import closing

class MyFlaskApp(Flask):
    # def read_posting_list(inverted, w):
    #     with closing(inverted_index_colab.MultiFileReader()) as reader:
    #         locs = inverted.posting_locs[w]
    #         s = str(locs[0][0])
    #         locs=[("C:\\Users\\HP\\Desktop\\postings_gcp\\"+s, locs[0][1])]
    #         b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
    #         posting_list = []
    #         for i in range(inverted.df[w]):
    #             doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
    #             tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
    #             posting_list.append((doc_id, tf))
    #         return posting_list

    def read_posting_list(self, w):
        inverted = self.inverted
        with closing(inverted_index_gcp.MultiFileReader()) as reader:
            locs = inverted.posting_locs[w]
            s = str(locs[0][0])
            locs=[("C:\\Users\\HP\\Desktop\\postings_gcp\\"+s, locs[0][1])]
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
    def get_top_100_pages(self, query):
        tfidf = {}
        for word in query:
            if word not in self.inverted.df.keys():
                continue
            # TODO - remove comment
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
                doc_len_normal = self.id_len_dict[doc_id]  # 4, 3, 10
                if doc_len_normal<100:
                    doc_len_normal *= 10000
                elif doc_len_normal > 1500:
                    doc_len_normal /= 10000
                elif doc_len_normal > 500:
                    doc_len_normal /= 1000
                elif doc_len_normal > 300:
                    doc_len_normal /= 100
                elif doc_len_normal > 50:
                    doc_len_normal /= 10

                # elif doc_len_normal > 700:
                #     doc_len_normal /= 100000

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

            best_100_id = list(tfidf.keys())[0:100]
            return best_100_id

    def get_id_title(self, id_list):
        titles = []
        for i in range(len(id_list)):
            titles.append(self.id_title_dict[id_list[i]])
        id_title = list(zip(id_list, titles))
        return id_title

    def run(self, host=None, port=None, debug=None, **options):

        # request.post(url="http://192.168.14.2:8080/get_pagerank", json=[1, 2, 3])
        # get_pagerank()
        self.inverted = inverted_index_gcp.InvertedIndex()
        self.corpus_size = 6348910
        self.id_len_dict = None
        self.id_title_dict = None
        self.id_page_rank_dict={}
        self.id_page_rank_dict2={}
        self.id_page_view_dict = {}
        self.id_page_view_dict2 = {}

        with open('C:\\Users\\HP\\Desktop\\project data\\page_views.pkl', 'rb') as f:
            data = pickle.load(f)
            self.id_page_view_dict = data
            # self.inverted.df = data.df
            # self.inverted.posting_locs = data.posting_locs


        # #self.inverted2 = inverted_index_gcp.InvertedIndex()
        # # load body index
        # with open('C:\\Users\\HP\\Desktop\\postings_gcp\\index.pkl', 'rb') as f:
        #     data = pickle.load(f)
        #     self.inverted.df = data.df
        #     self.inverted.posting_locs = data.posting_locs
        #
        # # load {id: len} dict
        # with open('C:\\Users\\HP\\Desktop\\project data\\docs_total_tokens.pkl', 'rb') as f:
        #     self.id_len_dict = pickle.load(f)
        #
        # # load {id: title} dict
        # with open('C:\\Users\\HP\\Desktop\\project data\\id_title_dict.pkl', 'rb') as f:
        #     self.id_title_dict = pickle.load(f)
        #     print()
        # # with open('C:\\Users\\HP\\Downloads\\postings_gcp_0_posting_locs.pickle', 'rb') as f:
        # #     data = pickle.load(f)
        # #     print(data)
        # #     self.inverted2.df=data.df
        # #     self.inverted2.posting_locs=data.posting_locs
        # # a= self.read_posting_list2('perry')

        # with open('C:\\Users\\HP\\Desktop\\project data\\id_page_rank.xls.csv', mode='r') as inp:
        #     reader = csv.reader(inp)
        #     print(reader)
        #     i = 0
        #     for row in reader:
        #         i+=1
        #         if i > 4000000:
        #             self.id_page_rank_dict2[row[0]]=row[1]
        #         else:
        #             self.id_page_rank_dict[row[0]]=row[1]


        #a = app.get_page_rank_by_id([1,2,3])
        #
        #         # break
        #     print()
        #     #self.mydict = dict((rows[0], rows[1]) for rows in reader)
        #     #print()

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

    def get_pageview(self, wiki_ids):
        res = []
        wiki_ids = [14673744, 24899468]
        for i in wiki_ids:
            i = str(i)
            if i in self.id_page_view_dict:
                res.append(self.id_page_view_dict[i])
            elif i in self.id_page_view_dict2:
                res.append(self.id_page_view_dict2[i])
        print(res)
        return res


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
    print(query)
    query = query.split()
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    # inv = app.inverted
    # query = list(query)
    x = app.get_top_100_pages(query)
    #print(x)
    y = app.get_id_title(x)
    # print(y)
    res = y
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
    # print(query)
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

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
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

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
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

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
    res = app.get_pageview(wiki_ids)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
    request.post(url="http://192.168.14.2:8080/get_pagerank", json=[1, 2, 3])

