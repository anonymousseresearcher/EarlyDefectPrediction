from __future__ import division, print_function
import sys, random, argparse

sys.dont_write_bytecode = True


class o():
    "Anonymous container"

    def __init__(i, **fields):
        i.override(fields)

    def override(i, d): i.__dict__.update(d); return i

    def __repr__(i):
        d = i.__dict__
        name = i.__class__.__name__
        return name + '{' + ' '.join([':%s %s' % (k, d[k])
                                      for k in i.show()]) + '}'

    def show(i):
        return [k for k in sorted(i.__dict__.keys())
                if not "_" in k]


The = o(cohen=0.3, small=3, epsilon=0.01,
        width=50, lo=0, hi=100, conf=0.01, b=1000, a12=0.56)

parser = argparse.ArgumentParser(
    description="Apply Scott-Knot test to data read from standard input")

p = parser.add_argument

p("--demo", default=False, action="store_true")
p("--cohen", type=float,
  default=0.3, metavar='N',
  help="too small if delta less than N*std of the data)")
p("--small", type=int, metavar="N", default=3,
  help="too small if hold less than N items")
p("--epsilon", type=float, default=0.01, metavar="N",
  help="a range is too small of its hi - lo < N")
p("--width", type=int, default=50, metavar="N",
  help="width of quintile display")
p("--text", type=int, default=12, metavar="N",
  help="width of text display")
p("--conf", type=float, default=0.01, metavar="N",
  help="bootstrap tests with confidence 1-n")
p("--a12", type=float, default=0.56, metavar="N",
  help="threshold for a12 test: disable,small,med,large=0,0.56,0.64,0.71")
p("--useA12", default=False, metavar="N",
  help="True if you want to use A12 instead of cliff's delta")
p("--latex", default=False, metavar="N",
  help="default is false and True for getting a latex table for the data")
p("--cdelta", default=0.147, metavar="N",
  help="value for cliff's delta to be considered not a small effect")

args = parser.parse_args()
The.cohen = args.cohen
The.small = args.small
The.epsilon = args.epsilon
The.conf = args.conf
The.width = args.width + 0
The.a12 = args.a12 + 0
The.text = args.text + 0
The.latex = args.latex
The.useA12 = args.useA12
The.cdelta = args.cdelta




def rdiv0():
    rdivDemo([
        ["x1", 0.34, 0.49, 0.51, 0.6],
        ["x2", 6, 7, 8, 9]])


"""
rank ,         name ,    med   ,  iqr 
----------------------------------------------------
   1 ,           x1 ,      51  ,    11 (*              |              ), 0.34,  0.49,  0.51,  0.51,  0.60
   2 ,           x2 ,     800  ,   200 (               |   ----   *-- ), 6.00,  7.00,  8.00,  8.00,  9.00


### Lesson One

Some similarities are obvious...

"""


def rdiv1():
    rdivDemo([
        ["x1", 0.1, 0.2, 0.3, 0.4],
        ["x2", 0.1, 0.2, 0.3, 0.4],
        ["x3", 6, 7, 8, 9]])


"""

rank ,         name ,    med   ,  iqr 
----------------------------------------------------
   1 ,           x1 ,      30  ,    20 (*              |              ), 0.10,  0.20,  0.30,  0.30,  0.40
   1 ,           x2 ,      30  ,    20 (*              |              ), 0.10,  0.20,  0.30,  0.30,  0.40
   2 ,           x3 ,     800  ,   200 (               |   ----   *-- ), 6.00,  7.00,  8.00,  8.00,  9.00

### Lesson Two

Many results often clump into less-than-many ranks.

"""


def rdiv2():
    rdivDemo([
        ["x1", 0.34, 0.49, 0.51, 0.6],
        ["x2", 0.6, 0.7, 0.8, 0.9],
        ["x3", 0.15, 0.25, 0.4, 0.35],
        ["x4", 0.6, 0.7, 0.8, 0.9],
        ["x5", 0.1, 0.2, 0.3, 0.4]])


"""
rank ,         name ,    med   ,  iqr 
----------------------------------------------------
   1 ,           x5 ,      30  ,    20 (---    *---    |              ), 0.10,  0.20,  0.30,  0.30,  0.40
   1 ,           x3 ,      35  ,    15 ( ----    *-    |              ), 0.15,  0.25,  0.35,  0.35,  0.40
   2 ,           x1 ,      51  ,    11 (        ------ *--            ), 0.34,  0.49,  0.51,  0.51,  0.60
   3 ,           x2 ,      80  ,    20 (               |  ----    *-- ), 0.60,  0.70,  0.80,  0.80,  0.90
   3 ,           x4 ,      80  ,    20 (               |  ----    *-- ), 0.60,  0.70,  0.80,  0.80,  0.90

### Lesson Three

Some results even clump into one rank (the great null result).

"""


def rdiv3():
    rdivDemo([
        ["x1", 101, 100, 99, 101, 99.5],
        ["x2", 101, 100, 99, 101, 100],
        ["x3", 101, 100, 99.5, 101, 99],
        ["x4", 101, 100, 99, 101, 100]])


"""

rank ,         name ,    med   ,  iqr 
----------------------------------------------------
   1 ,           x1 ,    10000  ,   150 (-------       *|              ),99.00, 99.50, 100.00, 101.00, 101.00
   1 ,           x2 ,    10000  ,   100 (--------------*|              ),99.00, 100.00, 100.00, 101.00, 101.00
   1 ,           x3 ,    10000  ,   150 (-------       *|              ),99.00, 99.50, 100.00, 101.00, 101.00
   1 ,           x4 ,    10000  ,   100 (--------------*|              ),99.00, 100.00, 100.00, 101.00, 101.00


#### Lesson Four

lesson four?

### Lesson Five

Some things had better clump to one thing (sanity check for the ranker).


"""


def rdiv5():
    rdivDemo([
        ["x1", 11, 11, 11],
        ["x2", 11, 11, 11],
        ["x3", 11, 11, 11]])


"""
rank ,         name ,    med   ,  iqr 
----------------------------------------------------
   1 ,           x1 ,    1100  ,     0 (*              |              ),11.00, 11.00, 11.00, 11.00, 11.00
   1 ,           x2 ,    1100  ,     0 (*              |              ),11.00, 11.00, 11.00, 11.00, 11.00
   1 ,           x3 ,    1100  ,     0 (*              |              ),11.00, 11.00, 11.00, 11.00, 11.00

### Lesson Six

Some things had better clump to one thing (sanity check for the ranker).


"""


def rdiv6():
    rdivDemo([
        ["x1", 11, 11, 11],
        ["x2", 11, 11, 11],
        ["x4", 32, 33, 34, 35]])


"""
rank ,         name ,    med   ,  iqr 
----------------------------------------------------
   1 ,           x1 ,    1100  ,     0 (*              |              ),11.00, 11.00, 11.00, 11.00, 11.00
   1 ,           x2 ,    1100  ,     0 (*              |              ),11.00, 11.00, 11.00, 11.00, 11.00
   2 ,           x4 ,    3400  ,   200 (               |          - * ),32.00, 33.00, 34.00, 34.00, 35.00

### Lesson Seven

All the above scales to succinct summaries of hundreds, thousands, millions of numbers

"""


def rdiv7():
    rdivDemo([
        ["x1"] + [rand() ** 0.5 for _ in range(256)],
        ["x2"] + [rand() ** 2 for _ in range(256)],
        ["x3"] + [rand() for _ in range(256)]
    ])





def _ab12():
    def a12slow(lst1, lst2):
        more = same = 0.0
        for x in sorted(lst1):
            for y in sorted(lst2):
                if x == y:
                    same += 1
                elif x > y:
                    more += 1
        return (more + 0.5 * same) / (len(lst1) * len(lst2))

    random.seed(1)
    l1 = [random.random() for x in range(5000)]
    more = [random.random() * 2 for x in range(5000)]
    l2 = [random.random() for x in range(5000)]
    less = [random.random() / 2.0 for x in range(5000)]
    for tag, one, two in [("1less", l1, more),
                          ("1more", more, less), ("same", l1, l2)]:
        t1 = msecs(lambda: a12(l1, less))
        t2 = msecs(lambda: a12slow(l1, less))
        print("\n", tag, "\n", t1, a12(one, two))
        print(t2, a12slow(one, two))




"""

Misc functions:

"""
rand = random.random
any = random.choice
seed = random.seed
exp = lambda n: math.e ** n
ln = lambda n: math.log(n, math.e)
g = lambda n: round(n, 2)


def median(lst, ordered=False):
    if not ordered: lst = sorted(lst)
    n = len(lst)
    p = n // 2
    if n % 2: return lst[p]
    q = p - 1
    q = max(0, min(q, n))
    return (lst[p] + lst[q]) / 2


def msecs(f):
    import time
    t1 = time.time()
    f()
    return (time.time() - t1) * 1000


def pairs(lst):
    "Return all pairs of items i,i+1 from a list."
    last = lst[0]
    for i in lst[1:]:
        yield last, i
        last = i


def xtile(lst, lo=The.lo, hi=The.hi, width=The.width,
          chops=[0.1, 0.3, 0.5, 0.7, 0.9],
          marks=["-", " ", " ", "-", " "],
          bar="|", star="*", show=" %3.0f"):
    """The function _xtile_ takes a list of (possibly)
    unsorted numbers and presents them as a horizontal
    xtile chart (in ascii format). The default is a
    contracted _quintile_ that shows the
    10,30,50,70,90 breaks in the data (but this can be
    changed- see the optional flags of the function).
    """

    def pos(p):
        return ordered[int(len(lst) * p)]

    def place(x):
        return int(width * float((x - lo)) / (hi - lo + 0.00001))

    def pretty(lst):
        return ', '.join([show % x for x in lst])

    ordered = sorted(lst)
    lo = min(lo, ordered[0])
    hi = max(hi, ordered[-1])
    what = [pos(p) for p in chops]
    where = [place(n) for n in what]
    out = [" "] * width
    for one, two in pairs(where):
        for i in range(one, two):
            out[i] = marks[0]
        marks = marks[1:]
    out[int(width / 2)] = bar
    out[place(pos(0.5))] = star
    return '(' + ''.join(out) + ")," + pretty(what)


def _tileX():
    import random
    random.seed(1)
    nums = [random.random() ** 2 for _ in range(100)]
    print(xtile(nums, lo=0, hi=1.0, width=25, show=" %5.2f"))


"""````

### Standard Accumulator for Numbers

Note the _lt_ method: this accumulator can be sorted by median values.

Warning: this accumulator keeps _all_ numbers. Might be better to use
a bounded cache.

"""


class Num:
    "An Accumulator for numbers"

    def __init__(i, name, inits=[]):
        i.n = i.m2 = i.mu = 0.0
        i.all = []
        i._median = None
        i.name = name
        i.rank = 0
        for x in inits: i.add(x)

    def s(i):
        return (i.m2 / (i.n - 1)) ** 0.5

    def add(i, x):
        i._median = None
        i.n += 1
        i.all += [x]
        delta = x - i.mu
        i.mu += delta * 1.0 / i.n
        i.m2 += delta * (x - i.mu)

    def __add__(i, j):
        return Num(i.name + j.name, i.all + j.all)

    def quartiles(i):
        def p(x): return float(g(xs[x]))

        i.median()
        xs = i.all
        n = int(len(xs) * 0.25)
        return p(n), p(2 * n), p(3 * n)

    def median(i):
        if not i._median:
            i.all = sorted(i.all)
            i._median = median(i.all)
        return i._median

    def __lt__(i, j):
        return i.median() < j.median()

    def spread(i):
        i.all = sorted(i.all)
        n1 = i.n * 0.25
        n2 = i.n * 0.75
        if len(i.all) <= 1:
            return 0
        if len(i.all) == 2:
            return i.all[1] - i.all[0]
        else:
            return i.all[int(n2)] - i.all[int(n1)]


"""

### Cliff's Delta

"""


def cliffsDelta(lst1, lst2, **dull):
    """Returns delta and true if there are more than 'dull' differences"""
    #    if not dull:
    #        dull = {'small': 0.147, 'medium': 0.33, 'large': 0.474} # effect sizes from (Hess and Kromrey, 2004)
    size = True
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j * repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j) * repeats
    d = (more - less) / (m * n)
    if abs(d) < The.cdelta:
        size = False
    return size


def lookup_size(delta, dull):
    """
    :type delta: float
    :type dull: dict, a dictionary of small, medium, large thresholds.
    """
    delta = abs(delta)
    if delta <= dull['small']:
        return False
    if dull['small'] < delta < dull['medium']:
        return True
    if dull['medium'] <= delta < dull['large']:
        return True
    if delta >= dull['large']:
        return True


def runs(lst):
    """Iterator, chunks repeated values"""
    for j, two in enumerate(lst):
        if j == 0:
            one, i = two, 0
        if one != two:
            yield j - i, one
            i = j
        one = two
    yield j - i + 1, two


"""

### The A12 Effect Size Test 

As above

"""


def a12slow(lst1, lst2):
    "how often is x in lst1 more than y in lst2?"
    more = same = 0.0
    for x in lst1:
        for y in lst2:
            if x == y:
                same += 1
            elif x > y:
                more += 1
    x = (more + 0.5 * same) / (len(lst1) * len(lst2))
    return x


def a12(lst1, lst2):
    "how often is x in lst1 more than y in lst2?"

    def loop(t, t1, t2):
        while t1.j < t1.n and t2.j < t2.n:
            h1 = t1.l[t1.j]
            h2 = t2.l[t2.j]
            h3 = t2.l[t2.j + 1] if t2.j + 1 < t2.n else None
            if h1 > h2:
                t1.j += 1;
                t1.gt += t2.n - t2.j
            elif h1 == h2:
                if h3 and h1 > h3:
                    t1.gt += t2.n - t2.j - 1
                t1.j += 1;
                t1.eq += 1;
                t2.eq += 1
            else:
                t2, t1 = t1, t2
        return t.gt * 1.0, t.eq * 1.0

    # --------------------------
    lst1 = sorted(lst1, reverse=True)
    lst2 = sorted(lst2, reverse=True)
    n1 = len(lst1)
    n2 = len(lst2)
    t1 = o(l=lst1, j=0, eq=0, gt=0, n=n1)
    t2 = o(l=lst2, j=0, eq=0, gt=0, n=n2)
    gt, eq = loop(t1, t1, t2)
    return gt / (n1 * n2) + eq / 2 / (n1 * n2) >= The.a12


def _a12():
    def f1(): return a12slow(l1, l2)

    def f2(): return a12(l1, l2)

    for n in [100, 200, 400, 800, 1600, 3200, 6400]:
        l1 = [rand() for _ in xrange(n)]
        l2 = [rand() for _ in xrange(n)]
        t1 = msecs(f1)
        t2 = msecs(f2)
        print(n, g(f1()), g(f2()), int((t1 / t2)))


"""
n   a12(fast)       a12(slow)       tfast / tslow
--- --------------- -------------- --------------
100  0.53           0.53               4
200  0.48           0.48               6
400  0.49           0.49              28
800  0.5            0.5               26
1600 0.51           0.51              72
3200 0.49           0.49             109
6400 0.5            0.5              244
````




"""


def sampleWithReplacement(lst):
    "returns a list same size as list"

    def any(n): return random.uniform(0, n)

    def one(lst): return lst[int(any(len(lst)))]

    return [one(lst) for _ in lst]


"""



"""


def testStatistic(y, z):
    """Checks if two means are different, tempered
     by the sample size of 'y' and 'z'"""
    tmp1 = tmp2 = 0
    for y1 in y.all: tmp1 += (y1 - y.mu) ** 2
    for z1 in z.all: tmp2 += (z1 - z.mu) ** 2
    s1 = (float(tmp1) / (y.n - 1)) ** 0.5
    s2 = (float(tmp2) / (z.n - 1)) ** 0.5
    delta = z.mu - y.mu
    if s1 + s2:
        delta = delta / ((s1 / y.n + s2 / z.n) ** 0.5)
    return delta


"""

 

"""


def bootstrap(y0, z0, conf=The.conf, b=The.b):
    """The bootstrap hypothesis test from
       p220 to 223 of Efron's book 'An
      introduction to the boostrap."""

    class total():
        "quick and dirty data collector"

        def __init__(i, some=[]):
            i.sum = i.n = i.mu = 0;
            i.all = []
            for one in some: i.put(one)

        def put(i, x):
            i.all.append(x);
            i.sum += x;
            i.n += 1;
            i.mu = float(i.sum) / i.n

        def __add__(i1, i2): return total(i1.all + i2.all)

    y, z = total(y0), total(z0)
    x = y + z
    tobs = testStatistic(y, z)
    yhat = [y1 - y.mu + x.mu for y1 in y.all]
    zhat = [z1 - z.mu + x.mu for z1 in z.all]
    bigger = 0.0
    for i in range(b):
        if testStatistic(total(sampleWithReplacement(yhat)),
                         total(sampleWithReplacement(zhat))) > tobs:
            bigger += 1
    return bigger / b < conf


"""

#### Examples

"""


def _bootstraped():
    def worker(n=1000,
               mu1=10, sigma1=1,
               mu2=10.2, sigma2=1):
        def g(mu, sigma): return random.gauss(mu, sigma)

        x = [g(mu1, sigma1) for i in range(n)]
        y = [g(mu2, sigma2) for i in range(n)]
        return n, mu1, sigma1, mu2, sigma2, \
               'different' if bootstrap(x, y) else 'same'

    # very different means, same std
    print(worker(mu1=10, sigma1=10,
                 mu2=100, sigma2=10))
    # similar means and std
    print(worker(mu1=10.1, sigma1=1,
                 mu2=10.2, sigma2=1))
    # slightly different means, same std
    print(worker(mu1=10.1, sigma1=1,
                 mu2=10.8, sigma2=1))
    # different in mu eater by large std
    print(worker(mu1=10.1, sigma1=10,
                 mu2=10.8, sigma2=1))


"""

Output:

"""

# _bootstraped()

(1000, 10, 10, 100, 10, 'different')
(1000, 10.1, 1, 10.2, 1, 'same')
(1000, 10.1, 1, 10.8, 1, 'different')
(1000, 10.1, 10, 10.8, 1, 'same')

"""



"""


def different(l1, l2):
    # return bootstrap(l1,l2) and a12(l2,l1)
    # return a12(l2,l1) and bootstrap(l1,l2)
    if The.useA12:
        return a12(l2, l1) and bootstrap(l1, l2)
    else:
        return cliffsDelta(l1, l2) and bootstrap(l1, l2)


"""


"""


def scottknott(data, cohen=The.cohen, small=The.small, useA12=The.a12 > 0, epsilon=The.epsilon):
    """Recursively split data, maximizing delta of
    the expected value of the mean before and
    after the splits.
    Reject splits with under 3 items"""
    all = reduce(lambda x, y: x + y, data)
    same = lambda l, r: abs(l.median() - r.median()) <= all.s() * cohen
    if useA12:
        same = lambda l, r: not different(l.all, r.all)
    big = lambda n: n > small
    return rdiv(data, all, minMu, big, same, epsilon)


def rdiv(data,  # a list of class Nums
         all,  # all the data combined into one num
         div,  # function: find the best split
         big,  # function: rejects small splits
         same,  # function: rejects similar splits
         epsilon):  # small enough to split two parts
    """Looks for ways to split sorted data,
    Recurses into each split. Assigns a 'rank' number
    to all the leaf splits found in this way.
    """

    def recurse(parts, all, rank=0):
        "Split, then recurse on each part."
        cut, left, right = maybeIgnore(div(parts, all, big, epsilon),
                                       same, parts)
        if cut:
            # if cut, rank "right" higher than "left"
            rank = recurse(parts[:cut], left, rank) + 1
            rank = recurse(parts[cut:], right, rank)
        else:
            # if no cut, then all get same rank
            for part in parts:
                part.rank = rank
        return rank

    recurse(sorted(data), all)
    return data


def maybeIgnore((cut, left, right), same, parts):
    if cut:
        if same(sum(parts[:cut], Num('upto')),
                sum(parts[cut:], Num('above'))):
            cut = left = right = None
    return cut, left, right


def minMu(parts, all, big, epsilon):
    """Find a cut in the parts that maximizes
    the expected value of the difference in
    the mean before and after the cut.
    Reject splits that are insignificantly
    different or that generate very small subsets.
    """
    cut, left, right = None, None, None
    before, mu = 0, all.mu
    for i, l, r in leftRight(parts, epsilon):
        if big(l.n) and big(r.n):
            n = all.n * 1.0
            now = l.n / n * (mu - l.mu) ** 2 + r.n / n * (mu - r.mu) ** 2
            if now > before:
                before, cut, left, right = now, i, l, r
    return cut, left, right


def leftRight(parts, epsilon=The.epsilon):
    """Iterator. For all items in 'parts',
    return everything to the left and everything
    from here to the end. For reasons of
    efficiency, take a first pass over the data
    to pre-compute and cache right-hand-sides
    """
    rights = {}
    n = j = len(parts) - 1
    while j > 0:
        rights[j] = parts[j]
        if j < n: rights[j] += rights[j + 1]
        j -= 1
    left = parts[0]
    for i, one in enumerate(parts):
        if i > 0:
            if parts[i]._median - parts[i - 1]._median > epsilon:
                yield i, left, rights[i]
            left += one


"""


## Putting it All Together

Driver for the demos:

"""


def rdivDemo(data, latex=False):
    def zzz(x):
        return int(100 * (x - lo) / (hi - lo + 0.00001))

    data = map(lambda lst: Num(lst[0], lst[1:]),
               data)
    print("rank,policy,median,iqr,q1,q2,q3,q4,q5")
    ranks = []
    for x in scottknott(data, useA12=True):
        ranks += [(x.rank, x.median(), x)]
    all = []

    for _, __, x in sorted(ranks): all += x.all
    all = sorted(all)
    lo, hi = all[0], all[-1]
    line = "----------------------------------------------------"
    last = None
    formatStr = '%%4s , %%%ss ,    %%s   , %%4s ' % The.text

    if latex:
        latexPrint(ranks, all)
    for _, __, x in sorted(ranks):
        q1, q2, q3 = x.quartiles()
        print((formatStr % \
               (x.rank + 1, x.name, q2, q3 - q1)) + \
              xtile(x.all, lo=lo, hi=hi, width=30, show="%5.2f"))
        last = x.rank


def _rdivs():
    seed(1)
    rdiv0();
    rdiv1();
    rdiv2();
    rdiv3();
    rdiv5();
    rdiv6();
    print("###");
    rdiv7()


def latexPrint(ranks, all):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%Latex Table%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("""\\begin{figure}[!t]
{\small
{\small \\begin{tabular}{llrrc}
\\arrayrulecolor{darkgray}
\\rowcolor[gray]{.9}  rank & treatment & median & IQR & \\\\""")
    lo, hi = all[0], all[-1]
    for _, __, x in sorted(ranks):
        q1, q2, q3 = x.quartiles()
        q1, q2, q3 = q1 * 100, q2 * 100, q3 * 100
        print("    %d &      %s &    %d &  %d & \quart{%d}{%d}{%d}{%d} \\\\" % (
        x.rank + 1, x.name.replace('_', '\_'), q2, q3 - q1, q1, q2, q3 - q1, q3 - q2))
        last = x.rank
    print("""\end{tabular}}
}
\\caption{%%%Enter Caption%%%
}\\label{fig:my fig}
\\end{figure}""")
    print("%%%%%%%%%%%%%%%%%%%%%%%End Latex Table%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


def xtile_modified(lst, lo=The.lo, hi=The.hi, width=The.width,
                   chops=[0.1, 0.3, 0.5, 0.7, 0.9],
                   marks=["-", " ", " ", "-", " "],
                   bar="|", star="*", show=" %3.0f"):
    """The function _xtile_ takes a list of (possibly)
    unsorted numbers and presents them as a horizontal
    xtile chart (in ascii format). The default is a
    contracted _quintile_ that shows the
    10,30,50,70,90 breaks in the data (but this can be
    changed- see the optional flags of the function).
    """

    def pos(p):
        return ordered[int(len(lst) * p)]

    def place(x):
        return int(width * float((x - lo)) / (hi - lo + 0.00001))

    def pretty(lst):
        return ', '.join([show % x for x in lst])

    ordered = sorted(lst)
    lo = min(lo, ordered[0])
    hi = max(hi, ordered[-1])
    what = [pos(p) for p in chops]
    where = [place(n) for n in what]
    out = [" "] * width
    for one, two in pairs(where):
        for i in range(one, two):
            out[i] = marks[0]
        marks = marks[1:]
    out[int(width / 2)] = bar
    out[place(pos(0.5))] = star
    print(what)
    return what


####################################

def thing(x):
    "Numbers become numbers; every other x is a symbol."
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x


def main():
    log = None
    latex = False#The.latex
    all = {}
    now = []
    for line in sys.stdin:
        for word in line.split():
            word = thing(word)
            if isinstance(word, str):
                now = all[word] = all.get(word, [])
            else:
                now += [word]
    rdivDemo([[k] + v for k, v in all.items()], latex)


if args.demo:
    _rdivs()
else:
    main()
