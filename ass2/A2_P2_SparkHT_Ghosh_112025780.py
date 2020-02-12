import sys
import csv
from pyspark import SparkContext
from pprint import pprint
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import stats


def scale(kvs):
    k = kvs[0]
    values = kvs[1]
    scaled_vs = StandardScaler().fit_transform(list(values))
    return k, scaled_vs


def linregv(kvs, n):
    k = kvs[0]
    values = kvs[1]
    y = []
    x = []
    for value in values:
        y.append(value[2])
        x.append(value[:n])

    y = np.matrix(y).reshape(-1, 1)
    x = np.matrix(x)
    ones = np.ones((x.shape[0], 1))
    x_ = np.hstack((x, ones))

    beta = np.linalg.pinv(np.dot(x_.T, x_)) * np.dot(x_.T, y)
    y_pred = np.matmul(x_, beta)

    return k, beta, y_pred, y, x_


def compute_pvalue(kvs, N):
    k, beta, y_pred, y, x_ = kvs
    df = y_pred.shape[0] - x_.shape[1] - 1
    ss = np.sum(np.square(np.subtract(y_pred, y)), axis=0) / df
    x_i = x_[:, 0]
    x_ss = np.var(x_i) * x_i.shape[0]
    t_value = abs(beta[0] / np.sqrt(ss / x_ss))
    p_value = 1 - stats.t.cdf(t_value, df=df)
    return k, (p_value[0][0] * N, beta.item((0, 0)))


def run_spark(word_path, county_path):
    sc = SparkContext()
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

    wordRDD = sc.textFile(word_path).mapPartitions(lambda line: csv.reader(line))
    countyRDD = sc.textFile(county_path).mapPartitions(lambda line: csv.reader(line))

    rdd1 = wordRDD.map(lambda x: (x[0], (x[1], x[3])))
    rdd2 = countyRDD.map(lambda x: (x[0], (x[23], x[24])))
    rdd3 = rdd1.join(rdd2)

    rdd_joined = rdd3.map(lambda x: (x[1][0][0], (x[1][0][1], x[1][1][0], x[1][1][1])))

    rdd_scaled = rdd_joined.groupByKey().map(lambda x: scale(x))

    N = rdd_scaled.count()
    print("Number of words", N)

    rdd_linreg1 = rdd_scaled.map(lambda x: linregv(x, 1)).persist()
    rdd_linreg2 = rdd_scaled.map(lambda x: linregv(x, 2)).persist()

    rdd_top1 = sc.parallelize(rdd_linreg1.sortBy(lambda x: x[1][0], ascending=False).take(20))
    rdd_bottom1 = sc.parallelize(rdd_linreg1.sortBy(lambda x: x[1][0]).take(20))

    rdd_top2 = sc.parallelize(rdd_linreg2.sortBy(lambda x: x[1][0], ascending=False).take(20))
    rdd_bottom2 = sc.parallelize(rdd_linreg2.sortBy(lambda x: x[1][0]).take(20))

    rdd_pvalue_top_1 = rdd_top1.map(lambda x: compute_pvalue(x, N))
    rdd_pvalue_bottom_1 = rdd_bottom1.map(lambda x: compute_pvalue(x, N))
    rdd_pvalue_top_2 = rdd_top2.map(lambda x: compute_pvalue(x, N))
    rdd_pvalue_bottom_2 = rdd_bottom2.map(lambda x: compute_pvalue(x, N))

    print("Top 20 words positively correlated with hd mortality")
    print("word,\t(pvalue_corrected,\tcoefficient)")
    pprint(rdd_pvalue_top_1.collect())

    print("Top 20 words negatively correlated with hd mortality")
    print("word,\t(pvalue_corrected,\tcoefficient)")
    pprint(rdd_pvalue_bottom_1.collect())
    print("Top 20 words positively correlated with hd mortality controlling for income")
    print("word,\t(pvalue_corrected,\tcoefficient)")
    pprint(rdd_pvalue_top_2.collect())
    print("Top 20 words negatively correlated with hd mortality controlling for income")
    print("word,\t(pvalue_corrected,\tcoefficient)")
    pprint(rdd_pvalue_bottom_2.collect())


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Invalid number of arguments")
        exit()

    wordPath = sys.argv[1]
    countyPath = sys.argv[2]
    run_spark(wordPath, countyPath)
