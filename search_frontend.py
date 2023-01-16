from flask import Flask, request, jsonify
from nltk.corpus import stopwords
import pandas as pd
import re
import math
from inverted_index_gcp import *
from collections import defaultdict
from gensim.models import KeyedVectors
import time

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became", "best"]
stopwords_frozen = frozenset(stopwords.words('english')).union(corpus_stopwords)


class MyFlaskApp(Flask):

    def run(self, host=None, port=None, debug=None, **options):
        # todo change links to the real one

        self.index_body = InvertedIndex.read_index('postings_body', 'index_body')
        self.index_title = InvertedIndex.read_index('postings_title', 'index_title')
        self.index_anchor = InvertedIndex.read_index('postings_anchor', 'index_anchor')
        self.all_docs_len_table = dict(pickle.load(open('doc_len.pickle', 'rb')))
        self.num_of_docs = len(self.all_docs_len_table)
        self.avg_len_docs = sum([int(val[1]) for val in
                                 self.all_docs_len_table.values()]) / self.num_of_docs  # avg of number words in doc for all the index
        self.page_rank = pd.read_csv('pageRank.csv', names=["doc_id", "page_rank"])
        self.page_views = dict(pickle.load(open('pageviews.pkl', 'rb')))
        temp = pickle.load(open('normal_values.pkl', 'rb'))
        self.normal_values = dict()

        for dict_tmp in temp:
            for key, val in dict_tmp.items():
                self.normal_values[str(key)] = val[1]

        self.word2vec = KeyedVectors.load('Model.model')

        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False



def add_word2vec(query_tokens):
    '''
    Take 4 Synonym words for each word of the query by using pretrained Word2Vec model, sort them and adding the top N
    words to the original query, which N <- len(query_tokens)

    Args:
        query_tokens: list of query tokens {str} after tokanization

    Returns: new list of tokens {str}

    '''
    new_query_token = []
    for term in query_tokens:
        try:
            terms_candidate = app.word2vec.wv.most_similar(term)
        except Exception:
            continue

        for i in range(4):
            new_query_token.append(terms_candidate[i])
    new_query_token = sorted(new_query_token, key=lambda x: x[1], reverse=True)[:len(query_tokens)]
    query_tokens = query_tokens * 3
    for tup in new_query_token:
        query_tokens.append(tup[0])
    return query_tokens


def merge_results_by_pageview(final_score, page_view, max_pv, score_rate=0.5, pv_rate=0.5):
    '''
    Merge two list of (doc_id,scores) , first list of final score and second is page view.
    Normalizing the score by its weight  , in page view case we normilized by the max pag view in ids range for getting
    score between 0-1. returning list of new score for each doc id
    Args:
        final_score: list of (doc_id,score) where score is tfidf score after merging title and body search
        page_view: list of (doc_id,num_page_view)
        max_pv: the maximum page_views of those doc ids
        score_rate: the weight we gave for the final score
        pv_rate: the weight we gave for page view

    Returns: dict items of tuples (doc_id,score)

    '''
    dic = defaultdict(float)
    for tup in final_score:
        dic[tup[0]] += tup[1] * score_rate
    for tup in page_view:
        dic[tup[0]] += tup[1] * pv_rate / max_pv
    return list(dic.items())


def second_round_check(final_score, query_tokens):
    '''
    The function takes the best answer from the search engine (and might take other answers that have very close rate for query),
    combine the titles with the given query,
    then search 5 more results with the words combination
    Args:
        final_score: list of best scores from the search engine
        query_tokens: query that was given to the engine- after tokenization

    Returns: 5 more results that could be relevant after finishing the first search - [tuple(doc_id,title)]

    '''
    title_second_check = []
    min_val = float(final_score[0][1] * 0.95)
    query_for_second_check = query_tokens * 3
    avg_diff = 0
    for tup in final_score:
        if tup[1] - min_val >= 0:
            title_second_check.append(tup)
        else:
            break
    title_second_check = change_to_title_second_check(title_second_check)
    for tup in title_second_check:
        for word in tokenize(tup[1]):
            query_for_second_check.append(word)

    res_second_check = get_resault_for_query_binary_with_counter(app.index_title, query_for_second_check, "title", 5)
    return change_to_title(res_second_check)


@app.route("/search")
def search():
    """ Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    start_time = time.time()
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    q = query.split(" ")
    query_tokens = tokenize(query)
    if len(query_tokens) == 0:
        return jsonify(res)
    original_query_token = query_tokens.copy()
    # for using word2Vec model
    # if len(q) / len(query_tokens) >= 2:
    #     query_tokens = add_word2vec(query_tokens)

    body_candidates = my_get_topN_score_for_queries(query_tokens, app.index_body, 2500)  # list 100 top scores by tf-idf
    final_score = body_candidates
    N = 200
    if len(q) <= 10 or len(final_score) == 0:
        title_score = get_resault_for_query_binary_with_counter(app.index_title, original_query_token, "title", 2500)
        if len(final_score) == 0 and len(title_score) == 0:
            return jsonify([])
        avg_title, avg_body = calc_average_helper(title_score), calc_average_helper(body_candidates)
        title_weight = avg_title / (avg_body + avg_title)
        body_weight = avg_body / (avg_body + avg_title)
        final_score = merge_results(title_score, body_candidates, len(original_query_token), title_weight, body_weight,N)

    page_view = list(map(lambda x: (x[0], get_page_view([x[0]])[0]), final_score[:N]))
    max_pv = max([val[1] for val in page_view])
    final_score = merge_results_by_pageview(final_score[:N], page_view[:N], max_pv, 0.75, 0.25)
    final_score = sorted(final_score, key=lambda x: x[1], reverse=True)[:30]
    res = change_to_title(final_score, "search")
    res_second_check = second_round_check(final_score, query_tokens)
    for tup in res_second_check:
        if tup not in res:
            res.append(tup)
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = body_index_search(query, 20)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.

    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    tokens = tokenize(query)
    if len(tokens) == 0:
        return jsonify(res)
    res = get_resault_for_query_binary_title(app.index_title, tokens, "title")
    res = list(map(lambda x: (x[0], app.all_docs_len_table[str(x[0])][0]), res))

    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    res = get_resault_for_query_binary_anchor(app.index_anchor, tokenize(query), "anchor")
    res = list(map(lambda x: (x[0], app.all_docs_len_table[str(x[0])][0]), res))  # changing score to title

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
    res = get_page_rank(wiki_ids)
    # END SOLUTION
    return jsonify(res)


def get_page_rank(wiki_ids):
    '''
    returns list of page rank for each wiki_ids, if id does not exist in the corpus we put 0 (zero) in the page rank
    Args:
        wiki_ids: list of ids {int}

    Returns:list of rank {int}

    '''
    res = []
    for ids in wiki_ids:
        row_number = app.page_rank[app.page_rank["doc_id"] == ids].index.values
        if len(row_number) != 0:
            res.append(app.page_rank.iloc[row_number[0]][1])
        else:
            res.append(0)
    return res


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

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
    res = get_page_view(wiki_ids)
    # END SOLUTION
    return jsonify(res)


def get_page_view(wiki_ids):
    """

    Args:
        wiki_ids: returns list of page view for each wiki_ids, if id does not exist in the corpus we put
        0 (zero) in the page view.

    Returns: list of page views correspond to the wiki ids.

    """
    res = [app.page_views.get(ids, 0) for ids in wiki_ids]
    return res


def calc_average_helper(score):
    '''
    calculate average of middle 1/3 arguments in the list.
    Args:
        score: list of results with doc_id and their rate for the given query -> [tuple(doc_id,rate))]

    Returns: the average of the middle third of the answers -> (int)

    '''
    final = 0
    length_title_candidate = int(len(score) / 3)
    if len(score) != 0:
        if length_title_candidate == 0:
            final = sum([val[1] for val in score]) / len(score)
        else:
            final = sum(
                [val[1] for val in score[length_title_candidate:2 * length_title_candidate]]) / length_title_candidate
    return final


def merge_results_body_bigger(smaller, bigger, smaller_w, bigger_w):
    '''
    merging two list of scores by giving weight for each list.
    Args:
        smaller: smaller list by elements -> list of (doc_id,score) shorter then the other
        bigger: longer list by elements
        smaller_w:lower weight
        bigger_w:higher weight

    Returns: merged dictionary ,key -> doc id ,value -> score after weighting

    '''
    dic = defaultdict(float)
    index = 0
    for i in range(len(smaller)):
        dic[smaller[i][0]] += smaller_w * smaller[i][1]
        dic[bigger[i][0]] += bigger_w * bigger[i][1]
        index = i
    for i in range(index, len(bigger)):
        dic[bigger[i][0]] += bigger_w * bigger[i][1]
    return dic


def merge_results(title_scores, body_scores, len_query, title_weight=0.5, text_weight=0.5, N=3):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: a list build upon the title index of  tuples representing scores as follows:(doc_id,score)

    body_scores: a list build upon the body/text index of  tuples representing scores as follows:(doc_id,score)

    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    list of  topN pairs as follows:(doc_id,score)

    """
    if len(title_scores) < len(body_scores):
        dic = merge_results_body_bigger(title_scores, body_scores, title_weight / len_query, text_weight, )
    else:
        dic = merge_results_body_bigger(body_scores, title_scores, text_weight, title_weight / len_query)
    return sorted(dic.items(), key=lambda x: x[1], reverse=True)[:N]


def calc_average_result(lst_to_convert):
    # calculate the average of the results with the first result appears twice
    avg_special = max([tup[1] for tup in lst_to_convert])
    num = 1
    for tup in lst_to_convert:
        avg_special += tup[1]
        num += 1
    return avg_special / num


def change_to_title(lst_to_convert, search_method="body"):
    """
    This function change score to title. if we in the search function we will filter the documents with the calc_average_result
    function. in other cases it will return the same list just with titles instead of score/
    Args:
        lst_to_convert: list of tuples (doc_id,score)
        search_method:identifier to know from where we got here

    Returns: new list of (doc_id,title) {int,int}

    """
    res = []
    avg_special = 0
    if search_method != "body":
        avg_special = calc_average_result(lst_to_convert)
    for tup in lst_to_convert:
        # change the score to the title of the page
        if tup[1] >= avg_special:
            title = app.all_docs_len_table[str(tup[0])][0]
            res.append((int(tup[0]), title))
    return res


def change_to_title_second_check(dict_to_change):
    res = []
    for tup in dict_to_change:
        # change the score to the title of the page
        title = app.all_docs_len_table[str(tup[0])][0]
        res.append((int(tup[0]), title))
    return res


def body_index_search(query, N=100):
    # return  top N list of (doc_id , title_name)
    q_tokens = tokenize(query)
    if len(q_tokens) == 0:
        return []
    tfidf_q = my_get_topN_score_for_queries(q_tokens, app.index_body, N)
    return change_to_title(tfidf_q, "body")


def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in stopwords_frozen]
    return list_of_tokens


def my_query_tfidf_vector(query_to_search, counter, index):
    '''
    this function calculate the tfidf for the tokens of the query.
    Args:
        query_to_search: list of query tokens
        counter: Counter of query
        index: inverted index we work with (body index in general)

    Returns: dictionary of tokens as keys and tfidf scores as values.

    '''
    dic = {}
    epsilon = .0000001
    for token in counter:
        if token in index.df.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log(app.num_of_docs / (df + epsilon), 10)  # smoothing
            dic[token] = tf * idf
    return dic


def get_posting_iter(index, query, typeIndex):  # body , title, anchor
    """
    This function returning two lists of words and postings lists for each term in the query


    Parameters:
    ----------
    index: inverted index
    query: list of query tokens
    typeIndex: the type of inverted index {str}
    ----------
    Returns: 2 lists, first words in documents where tokens of query appears, second list of postings lists respectively
    to the words list
    """
    if typeIndex == "anchor":
        dic = index.posting_lists_iter_for_anchor(query, typeIndex)
    else:
        dic = index.posting_lists_iter(query, typeIndex)
    return list(dic.keys()), list(dic.values())


def get_candidate_documents_and_scores(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """

    def calc_id_tfidf(doc_id, freq):
        return (doc_id, (freq / int(app.all_docs_len_table[str(doc_id)][1])) * math.log(
            app.num_of_docs / index.df[term], 10))

    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            normlized_tfidf = []
            for doc_id, freq in list_of_doc:
                id_tfidf = calc_id_tfidf(doc_id, freq)
                normlized_tfidf.append(id_tfidf)
            normlized_tfidf = sorted(normlized_tfidf, key=lambda tup: tup[0], reverse=True)
            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def get_top_n(sim_dict, N=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """

    return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


def my_get_topN_score_for_queries(query_to_search, index, N=3):
    '''
    in this function we calculate the cosine similarty for each document with the query.
    we chose our candidates and the calculate the sum of product for each term in the document and the term in the query
    after it we normalize the results. and returning the top N scores.
    Args:
        query_to_search: list of query tokens
        index: inverted index
        N:number of document to return

    Returns: sorted list of (doc_id,score)

    '''
    query_counter = Counter(query_to_search)
    Vector = my_query_tfidf_vector(query_to_search, query_counter, index)
    posting_words, posting_pls = get_posting_iter(index, query_to_search, "body")
    candidate_values = get_candidate_documents_and_scores(query_to_search, index, posting_words, posting_pls)

    dic_doc_id_final_score = defaultdict(float)
    for term in Vector.keys():
        for tup in candidate_values:
            dic_doc_id_final_score[tup[0]] += candidate_values[tup] * Vector[term]

    query_normalization = 0
    for term in query_counter:
        query_normalization += math.pow(query_counter[term], 2)
    query_normalization = 1 / math.sqrt(query_normalization)

    dic_doc_id_final_score = normalize_score(dic_doc_id_final_score, query_normalization)

    return get_top_n(dic_doc_id_final_score, N)


def normalize_score(dic_doc_id_final_score, query_normalization):
    '''
    divide the tfidf score by normalization factors that were calculated in advance.
    Args:
        dic_doc_id_final_score:  dict of doc_id and tfidf
        query_normalization: the normalization factor of the query

    Returns: dictionary of doc_id and score afternormalization

    '''
    for doc_id in dic_doc_id_final_score:
        noraml_value = app.normal_values[str(doc_id)]
        dic_doc_id_final_score[doc_id] = (dic_doc_id_final_score[doc_id]) * (
            query_normalization) / noraml_value
    return dic_doc_id_final_score


def get_resault_for_query_binary_anchor(index, query, typeIndex, N=-1):
    """
    query -> input query - list of tokens of the query (str)
    return -> list : tuples(doc_id,title) - sort by jacquard value calculation
    calculate method => union of query&title divided by the multiplicity of length of the query and length of the title
    """
    if len(query) == 0:
        return []
    words, pls = get_posting_iter(index, query, typeIndex)

    dic_jaccard = defaultdict(int)
    counter = Counter(query)
    for term in counter:
        if term in words:
            for doc_id in pls[words.index(term)]:
                if doc_id == 0: continue
                if str(doc_id) in app.all_docs_len_table.keys():
                    dic_jaccard[doc_id] += 1
    res = sorted(dic_jaccard.items(), key=lambda x: x[1], reverse=True)
    if N != -1:
        res = res[:N]

    return res


def get_resault_for_query_binary_title(index, query, typeIndex, N=-1):
    """
    query -> input query - list of tokens of the query (str)
    return -> list : tuples(doc_id,title) - sort by jacquard value calculation
    calculate method => union of query&title divided by the multiplicity of length of the query and length of the title
    """
    if len(query) == 0:
        return []
    words, pls = get_posting_iter(index, query, typeIndex)
    dic_appearance = defaultdict(int)
    counter = Counter(query)

    for term in counter:
        if term in words:
            for tup in pls[words.index(term)]:
                if tup[0] == 0: continue
                dic_appearance[tup[0]] += 1

    res = sorted(dic_appearance.items(), key=lambda x: x[1], reverse=True)
    if N != -1:
        res = res[:N]
    return res


def get_resault_for_query_binary_with_counter(index, query, typeIndex, N=-1):
    """
    query -> input query - list of tokens of the query (str)
    return -> list : tuples(doc_id,title) - sort by jacquard value calculation
    calculate method => union of query&title divided by the multiplicity of length of the query and length of the title
    """

    words, pls = get_posting_iter(index, query, typeIndex)
    dic_appearance = defaultdict(float)
    counter = Counter(query)
    for term in counter:
        if term in words:
            for tup in pls[words.index(term)]:
                dic_appearance[tup[0]] += 1 * counter[term]

    res = sorted(dic_appearance.items(), key=lambda x: x[1], reverse=True)

    if N != -1:
        res = res[:N]

    return res


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
