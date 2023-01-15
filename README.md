# wiki Search Engine
code structure:
search_frontend.py - initiallization of the search engine on the VM machine over GCP

inverted_index_gcp.py - indexer of documents which creates postings list for the search engine over GCP

functionality:
search()- based on body and title search and merge with page views doc, returns up to a 100 of the best search results for the query

get_pagerank()-Returns PageRank values for a list of provided wiki article IDs

get_pageview() - Returns the number of page views that each of the provide wiki articles had

get_resault_for_query_binary_anchor(index, query, typeIndex, N=-1) - implementation of anchor search

get_resault_for_query_by_title(index, query, typeIndex, N=-1) - implementation of title search

my_get_topN_score_for_queries(query_to_search, index, N=3) - implementation of body search with cosine similarity


