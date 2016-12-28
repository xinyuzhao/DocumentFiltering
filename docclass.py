import re
import math

splitter = re.compile(r'\W*')

def getWords(doc):
    words = set(s.lower() for s in splitter.split(doc) 
            if len(s) > 1 and len(s) < 20)

    return words

def sampleTrain(cl):
    cl.train('Nobody owns the water.','good')
    cl.train('the quick rabbit jumps fences','good')
    cl.train('buy pharmaceuticals now','bad')
    cl.train('make quick money at the online casino','bad')
    cl.train('the quick brown fox jumps','good')


class classifier:
    def __init__(self, getFeatures, filename = None):
        # Counts of feature/category combinations
        self.fc = {}
        # Counts of documents in each category
        self.cc = {}
        # Threshold for each category
        self.thres = {}

        self.getFeatures = getFeatures

    # Increase the count of a feature/category pair
    def incf(self, f, cat):
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat] += 1

    # increase the count of a category
    def incc(self, cat):
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1
        
    # The number of times a feature has appeared in a category
    def fcount(self, f, cat):
        if f in self.fc and cat in self.fc[f]:
            return float(self.fc[f][cat])

        return 0.0
    
    # The number of items in a category
    def catCount(self, cat):
        if cat in self.cc:
            return float(self.cc[cat])
        
        return 0

    # The total number of items
    def totalCount(self):
        return sum(self.cc.values())

    # The list of all categories
    def categories(self):
        return self.cc.keys()

    def setThreshold(self, cat, t):
        self.thres[cat] = t

    def getThreshold(self, cat):
        if cat not in self.thres:
            return 1.0
        
        return self.thres[cat]

    # Compute conditional probability P(f|cat)
    def fprob(self, f, cat, weight=1.0, ap=0.5):
        if self.catCount(cat) == 0:
            basicProb = 0
        else:
            basicProb = self.fcount(f, cat) / self.catCount(cat)

        total = sum([self.fcount(f, c) for c in self.categories()])

        return (weight * ap + total * basicProb) / (total + weight) 

    def train(self, item, cat):
        features = self.getFeatures(item)

        for f in features:
            self.incf(f, cat)

        self.incc(cat)

class naiveBayes(classifier):
    def docProb(self, item, cat):
        features = self.getFeatures(item)

        p = 1
        for f in features:
            p *= self.fprob(f, cat)

        return p

    def prob(self, item, cat):
        catprob = self.catCount(cat)/self.totalCount()
        docprob = self.docProb(item, cat)

        return catprob * docprob

    def classify(self, item, default=None):
        probs = {}

        max = 0.0
        for cat in self.categories():
            probs[cat] = self.prob(item, cat)
            if probs[cat] > max:
                max = probs[cat]
                best = cat

        for cat in probs:
            if cat == best:
                continue
            if probs[cat] * self.getThreshold(best) > probs[best]:
                return default

            return best





