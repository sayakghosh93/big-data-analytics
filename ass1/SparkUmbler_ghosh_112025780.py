# name, id

# Template code for CSE545 - Spring 2019
# Assignment 1 - Part II
# v0.01
from pyspark import SparkContext
import mmh3
import math
import re

location_filter = None
Lsmall = 50
Llarge = 2000
fl_hash_count = 3

nonfluencies_regex = {"MM": "mm+",
                      "OH": "oh+|ah+",
                      "SIGH": "sigh|sighed|sighing|sighs|ugh|uh",
                      "UM": "umm*|hmm*|huh"}


def generateKeyValue(x):
    def parseLabel(label):
        parsed = label[0].split(':')
        asize = (parsed[1].split(',')[0], parsed[1].split(',')[1])
        bsize = (parsed[2].split(',')[0], parsed[2].split(',')[1])

        return parsed[0], asize, bsize, label[1], label[2]

    matrix, asize, bsize, i, j = parseLabel(x[0])
    kvs = []
    if matrix == 'A':
        krange = int(bsize[1])
        for jin in range(krange):
            kvs.append(((i, jin), (j, matrix, x[1])))
    else:
        krange = int(asize[0])
        for jin in range(krange):
            kvs.append(((jin, j), (i, matrix, x[1])))

    return kvs


def reduceMultiply(kvs):
    k = kvs[0]
    vs = kvs[1]
    sum = 0
    alist = []
    blist = []
    for x in vs:
        if x[1] == 'A':
            alist.append(x)
        else:
            blist.append(x)
    alist.sort()
    blist.sort()

    index = 0

    while (index < len(alist) and index < len(blist)):
        if alist[index][0] == blist[index][0]:
            sum += alist[index][2] * blist[index][2]
        index += 1
    return (k, sum)


def sparkMatrixMultiply(rdd):
    # rdd where records are of the form:
    # returns an rdd with the resulting matrix

    # resultRdd = rdd  # REVISE TO COMPLETE
    rdd1 = rdd.flatMap(lambda x: generateKeyValue(x))
    rdd2 = rdd1.groupByKey()
    resultRdd = rdd2.map(lambda x: reduceMultiply(x))

    return resultRdd


def getBitValue(val, i, L):
    hit = mmh3.hash(val, i) % L
    bitarr = ["0"] * L
    posn = 0
    while hit != 0 and posn < L:
        bit = hit % 2
        bitarr[posn] = str(bit)
        hit = hit / 2
        posn += 1
    return "".join(bitarr)[::-1]


def mapTweet(tweet):
    global nonfluencies_regex
    tweet_tokens = tweet.split(' ')
    punctuations = ['!,', '.', '?', '-']
    length = len(tweet_tokens)
    res = []
    count = 0

    for i, tweet_token in enumerate(tweet_tokens):
        for nonfluency, regex in nonfluencies_regex.items():
            if re.match(regex, tweet_token):
                for j in range(1, length - i):
                    if count == 3:
                        break
                    if i + j < length and tweet_tokens[i + j] not in punctuations:
                        res.append(tweet_tokens[i + j])
                        count += 1
                resStr = " ".join(res)

                return nonfluency, resStr
    return " ", " "


def validLocation(row):
    global location_filter
    if location_filter.check(row[0]):
        return True
    return False


def countDistinct(x, L):
    k = x[0]
    vs = x[1]
    maxTrailingZero = 0
    for i in range(fl_hash_count):
        for v in vs:
            bitval = getBitValue(v, i, L)
            trailingZero = len(bitval.split("1")[-1])
            if trailingZero > maxTrailingZero and trailingZero != L:
                maxTrailingZero = trailingZero

    return k, 2 ** maxTrailingZero


def umbler(sc, rdd, L):
    # sc: the current spark context
    #    (useful for creating broadcast or accumulator variables)
    # rdd: an RDD which contains location, post data.
    #
    # returns a *dictionary* (not an rdd) of distinct phrases per um category

    # SETUP for streaming algorithms

    # PROCESS the records in RDD (i.e. as if processing a stream:
    # a foreach transformation
    filteredAndCountedRdd = rdd  # REVISE TO COMPLETE

    #  (will probably require multiple lines and/or methods)

    # return value should fit this format:

    distinctPhraseCounts = {'MM': 0,
                            'OH': 0,
                            'SIGH': 0,
                            'UM': 0}
    rdd1 = rdd.filter(validLocation).map(lambda x: mapTweet(x[1].lower()))
    rdd2 = rdd1.groupByKey()
    rdd3 = rdd2.map(lambda x: countDistinct(x, L))

    for key, val in rdd3.collect():
        if key != " ":
            distinctPhraseCounts[key] = val
    return distinctPhraseCounts


################################################
## Testing Code (subject to change for testing)

import numpy as np
from pprint import pprint
from scipy import sparse


def createSparseMatrix(X, label):
    sparseX = sparse.coo_matrix(X)
    list = []
    for i, j, v in zip(sparseX.row, sparseX.col, sparseX.data):
        list.append(((label, i, j), v))
    return list


class BloomFilter(object):
    def __init__(self, items_count, fp_prob):

        # False posible probability in decimal
        self.fp_prob = fp_prob

        # Size of bit array to use
        self.size = self.get_size(items_count, fp_prob)

        # number of hash functions to use
        self.hash_count = self.get_hash_count(self.size, items_count)

        # Bit array of given size
        self.bit_array = [False] * (self.size)

        # initialize all bits as 0
        # self.bit_array.setall(0)

    def add(self, item):

        for i in range(self.hash_count):
            hit = mmh3.hash(item, i) % self.size
            self.bit_array[hit] = True

    def check(self, item):
        for i in range(self.hash_count):
            hit = mmh3.hash(item, i) % self.size
            if self.bit_array[hit] == False:
                return False
        return True

    @classmethod
    def get_size(self, n, p):
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    @classmethod
    def get_hash_count(self, m, n):
        k = (m / n) * math.log(2)
        return int(k)


def runTests(sc):
    # runs MM and Umbler Tests for the given sparkContext

    # MM Tests:
    print("\n*************************\n MatrixMult Tests\n*************************")
    test1 = [(('A:2,1:1,2', 0, 0), 2.0), (('A:2,1:1,2', 1, 0), 1.0), (('B:2,1:1,2', 0, 0), 1), (('B:2,1:1,2', 0, 1), 3)]
    test2 = createSparseMatrix([[1, 2, 4], [4, 8, 16]], 'A:2,3:3,3') + createSparseMatrix(
        [[1, 1, 1], [2, 2, 2], [4, 4, 4]], 'B:2,3:3,3')
    test3 = createSparseMatrix(np.random.randint(-10, 10, (10, 100)), 'A:10,100:100,12') + createSparseMatrix(
        np.random.randint(-10, 10, (100, 12)), 'B:10,100:100,12')
    mmResults = sparkMatrixMultiply(sc.parallelize(test1))
    pprint(mmResults.collect())
    mmResults = sparkMatrixMultiply(sc.parallelize(test2))
    pprint(mmResults.collect())
    mmResults = sparkMatrixMultiply(sc.parallelize(test3))
    pprint(mmResults.collect())

    # Umbler Tests:
    print("\n*************************\n Umbler Tests\n*************************")
    testFileSmall = 'publicSampleLocationTweet_small.csv'
    testFileLarge = 'publicSampleLocationTweet_large.csv'

    # setup rdd
    import csv
    import sys
    csv.field_size_limit(sys.maxsize)

    smallTestRdd = sc.textFile(testFileSmall, use_unicode=False).map(
        lambda x: x.decode("ascii", "ignore")).mapPartitions(
        lambda line: csv.reader(line))

    location_count = sum(1 for line in open('umbler_locations.csv'))
    global location_filter
    location_filter = BloomFilter(location_count, 0.05)

    with open('umbler_locations.csv') as inputFile:
        lines = inputFile.readlines()
        for line in lines:
            location_filter.add(line)

    pprint(smallTestRdd.take(5))  # uncomment to see data
    pprint(umbler(sc, smallTestRdd, Lsmall))

    largeTestRdd = sc.textFile(testFileLarge, use_unicode=False).map(
        lambda x: x.decode("ascii", "ignore")).mapPartitions(lambda line: csv.reader(line))
    pprint(largeTestRdd.take(5))  # uncomment to see data
    pprint(umbler(sc, largeTestRdd, Llarge))

    return


sc = SparkContext("local", "app name")
runTests(sc)
