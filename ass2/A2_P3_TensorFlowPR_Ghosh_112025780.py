import sys
import numpy as np
import tensorflow as tf


def print_ranks(rank_vector):
    rank_idx_dec = np.argsort(rank_vector, axis=0)[::-1][:20]
    rank_idx_asc = np.argsort(rank_vector, axis=0)[:20]

    print("Top 20 pages")
    print("Node\tRank")
    for idx in rank_idx_dec:
        print(idx[0], rank_vector[idx][0][0], sep='\t')

    print("Bottom 20 pages")
    print("Node\tRank")
    for idx in rank_idx_asc:
        print(idx[0], rank_vector[idx][0][0], sep='\t')


def run_pagerank(input_file):
    pages = np.loadtxt(input_file, dtype=int)
    surfer_prob = 0.15
    threshold = 0.001
    beta = 1 - surfer_prob

    pages[:, 0], pages[:, 1] = pages[:, 1].copy(), pages[:, 0].copy()

    N = max(max(pages[:, 0]), max(pages[:, 1])) + 1
    nrow, ncol = N, N

    M = tf.sparse.placeholder(dtype=tf.float64, shape=[nrow, ncol])
    M_ = M / tf.sparse.reduce_sum(M, axis=0)
    surfer_matrix = np.ones([nrow, 1]) * (surfer_prob / int(nrow))
    rank_vector = np.ones([nrow, 1]) * (1. / nrow)

    print("rank vector", rank_vector)

    v = tf.placeholder(tf.float64, [ncol, 1])
    pagerank_vector_1 = tf.sparse.sparse_dense_matmul(M_, v) * beta
    pagerank_vector = tf.add(pagerank_vector_1, surfer_matrix)

    change_from_iteration = tf.linalg.norm(
        tf.subtract(pagerank_vector, v), ord=1)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        error = float('inf')
        while (error > threshold):
            new_rank_vector = sess.run(pagerank_vector, feed_dict={
                M: tf.SparseTensorValue(indices=pages, values=np.ones([pages.shape[0]]),
                                        dense_shape=[nrow, ncol]), v: rank_vector})
            error = sess.run(change_from_iteration, feed_dict={pagerank_vector: new_rank_vector, v: rank_vector})
            rank_vector = new_rank_vector
            print("vector", rank_vector)
            print("change", error)
    print_ranks(rank_vector)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Invalid number of arguments")
        exit(1)

    input_file = sys.argv[1]
    run_pagerank(input_file)
