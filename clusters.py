from math import sqrt
import random


def readfile(filename):
    '''
    The file has columns of the words count in different blogs
    The first column has the titles of the blogs while the rest have the
    word counts of each word in the blogs
    The rows hold the blogs and the count of words
    The first row has the words while the rest have the counts of the words
    in the blogs
    The files looks like this:
    Blogs word1 word2 word3 word4
    blog1 10    11    3     6
    blog2 4     8     20    2
    blog  9     2     6     1
    '''
    lines = [line for line in file(filename)]
    colnames = lines[0].strip().split('\t')[1:]  # holds the words in the blogs
    rownames = []  # will hold the blog titles
    data = []  # the counts of words in blogs
    for line in lines[1:]:
        p = line.strip().split('\t')
        rownames.append(p[0])
        data.append([float(x) for x in p[1:]])
    return rownames, colnames, data


def pearson(v1, v2):
    '''Returns the similarity between v1 and v2.

    1.0 means very similar and 0.0 means no correlation. -1.0 means
    anticorrelation.  v1 and v2 must have the same number of elements.'''
    n = len(v1)
    if n == 0:
        return 0

    sum1 = sum(v1)
    sum2 = sum(v2)

    sqSum1 = sum([pow(v, 2) for v in v1])
    sqSum2 = sum([pow(v, 2) for v in v2])

    pSum = sum([v1[i] * v2[i] for i in range(n)])

    num = pSum - (sum1*sum2/n)
    den = sqrt((sqSum1 - pow(sum1, 2)/n) * (sqSum2 - pow(sum2, 2)/n))
    if den == 0:
        return 0

    return 1.0 - num/den  # closely related items will have a smaller distance between them


class bicluster(object):
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
        '''
        A cluster
        vect is the data of the cluster
        left is the left member of the cluster
        right is the right member of the cluster
        distance is the distance between the members of the cluster
        id is the id of the cluster
        '''
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.id = id

    def __eq__(self, b):
        return (self.vec == b.vec
            and self.left == b.left
            and self.right == b.right
            and self.distance == b.distance
            and self.id == b.id)

    # If we have __eq__, we better have __ne__ too
    # so that `not (a == b) == a != b`
    def __ne__(self, b):
        return not (self == b)

    # If we have __eq__, we better have __hash__ too
    # so that `a == b => hash(a) == has(b)`. Since we don't need bicluster objects
    # as dict keys, it's ok if this function fails loudly (instead of silently
    # returning a wrong value, which is the defaul)
    def __hash__(self):
        raise NotImplementedError

    def __str__(self):
        return '%s %f %d (%s %s)' % (str(self.vec), self.distance, self.id,
            self.left, self.right)


def hcluster(rows, distance=pearson):
    distances = {}
    currentclustid = -1

    # Clusters start off as just blogs, which are the rows
    clust = [bicluster(rows[i], id=i) for i in xrange(len(rows))]

    # O(n^3), yuck! Effectively, only the distance() calls are expensive,
    # and we cache them, so this is really O(n^2)
    while len(clust) > 1:
        # starting with the first 2 elements of the cluster as the closest pair
        lowestpair = 0, 1
        closest = distance(clust[0].vec, clust[1].vec)

        # Loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i + 1, len(clust)):
                # cache distances. Makes this much faster.
                # (can't use the cache() function because we cache on ids, not
                # function arguments. as clust shrinks, we can't just cache on indices
                # either)
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(
                        clust[i].vec, clust[j].vec)
                d = distances[(clust[i].id, clust[j].id)]
                
                if d < closest:
                    closest = d
                    lowestpair = i, j

        # Merge closest pair into a single vector by doing the average of the
        # similar words occurrence
        mergevec = [(clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i]) / 2.0
            for i in range(len(clust[0].vec))]

        # creating a new cluster from the merged clusters
        newcluster = bicluster(mergevec, left=clust[lowestpair[0]],
            right=clust[lowestpair[1]], distance=closest, id=currentclustid)

        # Update: the merged clusters need to be removed and the merge
        # needs to be added to the list of the clusters
        # Need to del() bigger index first! Deletion in a list leads to
        # shifting of the reamaining members to take the lower positions if
        # they are available
        currentclustid -= 1  # cluster ids that were not in the original set are negative
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)

    return clust[0]


# printin the clusters
def printclust(clust, labels=None, n=0):
    print ' ' * n,
    if clust.id < 0:  # branch
        print '-'
    else:
        print labels[clust.id] if labels else clust.id

    if clust.left:
        printclust(clust.left, labels=labels, n=n+1)
    if clust.right:
        printclust(clust.right, labels=labels, n=n+1)


def rotate_matrix(data):
    '''
    converting the rows to columns and columns to rows
    '''
    new_data = []
    for i in len(data[0]):
        new_row = [data[j][i] for j in range(len(data))]
        new_data.append(new_row)

    return new_data


def transpose(data):
  return map(list, zip(*data))


def kcluster(rows, distance=pearson, k=4):
    """Returns a list of `k` lists, each containing all indices of a cluster.
    rows - The data to be clustered
    distance - the method of calculating the distance
    k - the number of clusters needed"""

    # the 2 lines below are used to find the k centroids
    # the number of data points in each centroid is the same as the number of
    # data points in each row of data
    # the values of the initial centroids are obtained by finding random
    # values between the max and the min of a datapoint in the rows
    ranges = [(min([row[i] for row in rows]), max([row[i] for row in rows])) for i in range(len(rows[0]))]
    clusters = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0] for i in range(len(rows[0]))] for j in range(k)]

    lastmatches = None
    for t in range(100):  # will iterate to a max of 100 times
        print 'Iteration', t
        # will hold lists of row indixes belonging to a cluster
        bestmatches = [[] for i in range(k)]

        # find best centroid for each row i.e the centroid closest to
        # each row. It is the centroid with the lowest distance from
        # the row
        for j in range(len(rows)):
            row = rows[j]
            bestmatch = 0
            for i in range(k):
                d = distance(clusters[i], row)
                if d < distance(clusters[bestmatch], row):
                    bestmatch = i

            bestmatches[bestmatch].append(j)
          
        # if the results didn't change in this iteration, we are done
        # because we cannot obtain a better classification than this. The
        # learning has already saturated
        if bestmatches == lastmatches:
            break
        lastmatches = bestmatches

        # move centroids to the averages of their elements
        # this is done by first adding all the data points of rows belonging
        # to a cluster and then dividing by the number of rows that
        # belong to the cluster
        for i in range(k):
            avgs = [0.0] * len(rows[0])
            if bestmatches[i] > 0:
                for rowid in bestmatches[i]:
                    for m in rows[rowid]:
                        avgs[m] = rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])

    return bestmatches


def tanimoto_dist(v1, v2):
  c1, c2, shr = 0, 0, 0
  for i in range(len(v1)):
    if v1[i] != 0: c1 += 1
    if v2[i] != 0: c2 += 1
    if v1[i] != 0 and v2[i] != 0: shr += 1
  return 1.0 - float(shr)/(c1 + c2 - shr)


def hypot(v):
  return sqrt(sum([x*x for x in v]))


def euclid_dist(v1, v2):
  return hypot([v[0] - v[1] for v in zip(v1, v2)])


def scaledown(data, distance=pearson_dist, rate=0.01):
  n = len(data)

  realdist = [[distance(data[i], data[j]) for j in range(n)] for i in range(n)]
  outersum = 0.0

  # random start positions
  loc = [[random.random(), random.random()] for i in range(n)]

  lasterror = None
  for m in range(0, 1000):
    # find projected distance
    fakedist = [[euclid_dist(loc[i], loc[j])
      for j in range(n)] for i in range(n)]

    # move points
    grad = [[0.0, 0.0] for i in range(n)]

    totalerror = 0
    for k in range(n):
      for j in range(n):
        if j == k: continue

        # error is percent difference between distances
        errorterm = (fakedist[j][k] - realdist[j][k])/realdist[j][k]

        grad[k][0] += ((loc[k][0] - loc[j][0])/fakedist[j][k]) * errorterm
        grad[k][1] += ((loc[k][1] - loc[j][1])/fakedist[j][k]) * errorterm

        totalerror += abs(errorterm)
    print totalerror

    # if we got worse by moving the points, quit
    if lasterror and lasterror < totalerror: break

    # also break if the improvement is only very small
    if lasterror and lasterror - totalerror < 1e-15: break

    lasterror = totalerror

    # move points by learning rate times gradient
    if k in range(n):
      loc[k][0] -= rate * grad[k][0]
      loc[k][1] -= rate * grad[k][1]

  return loc


if __name__ == '__main__':
  # stupid demo
  import drawclust
  blognames, words, data = readfile('blogdata.txt')
  c = hcluster(data)
  #printclust(c, labels=blognames)
  drawclust.drawdendogram(c, blognames, 'dendo.png')
  print 'Wrote dendo.png'

  ## this is _much_ slower, as hcluster computes O(rows^2) many distances,
  ## and there are many more words than blognames in out data.
  #c = hcluster(transpose(data))
  #drawclust.drawdendogram(c, words, 'dendo_words.png')
  #print 'Wrote dendo_words.png'

  kclust = kcluster(data, k=10)
  for i in range(len(kclust)):
    print 'k-cluster %d:' % i, [blognames[r] for r in kclust[i]]
    print

  # another demo
  coords = scaledown(data)
  drawclust.draw2d(coords, blognames, filename='blogs2d.png')
  print 'Wrote blogs2d.png'

  # and yet another demo
  wants, people, data = readfile('official_zebo.txt')
  cl = hcluster(data, distance=tanimoto_dist)
  drawclust.drawdendogram(cl, wants, 'wants.png')
  print 'Wrote wants.png'
