import numpy as np

def gini_inpurity(X,y,indices,num_class):
    X_ = X[indices]
    y_ = y[indices]
    return 1 - sum((len(y_[y_ == i])/len(indices))**2 for i in range(num_class))

class DTNode:
    def __init__(self,par=None,indices=None,depth=0):
        self.depth = depth
        self.indices = indices
        self.par = par
        self.left = None
        self.right = None
        self.val = None
        self.rule = None
        self.label = None

class DecisionTree:
    def __init__(self, max_depth=2):
        self.root = DTNode()
        self.max_depth = max_depth
        
    def fit(self, X, y):
        self.X, self.y,self.num_class,self.root.indices = X,y,len(set(y)),list(range(len(y)))
        now = self.root
        path = [] #デバッグ用。現在のノードに至るまでの過程を記録
        count = 0
        while True:
            #print(path)
            if now.depth < self.max_depth and now.val == None:
                now.left, now.right = DTNode(par=now,depth=now.depth+1), DTNode(par=now,depth=now.depth+1)
                self._fit_single(now)
            if now.depth < self.max_depth - 1 and (self._check(now.left) and now.left.val == None):
                now = now.left
                path.append(0)
            elif now.depth < self.max_depth - 1 and (self._check(now.right) and now.right.val == None):
                now = now.right
                path.append(1)
            elif now == self.root: return
            else:
                now = now.par
                path.pop()
            
            if now == self.root:
                count += 1
                
            if count == 2: return
        
    # 同じラベルかどうかチェックする
    def _check(self, node):
        res = [self.y[i] for i in node.indices]
        return len(set(res)) != 1
        
    def _fit_single(self, node):
        node.val, node.rule, node.left.indices, node.right.indices = self._separate(node.indices)
        node.left.label = np.argmax(np.bincount(self.y[node.left.indices]))
        node.right.label = np.argmax(np.bincount(self.y[node.right.indices]))
        
    # 本体の識別部分
    def _separate(self, indices):
        MIN = 1
        argmin = -1
        for j in range(self.X.shape[1]):
            inds = {}
            for i in indices:
                x = self.X[i,j]
                if x in inds.keys(): inds[x].append(i)
                else: inds[x] = [i]
            keys = sorted(inds.keys())
            id1,id2 = set(),set(indices)
            if argmin == -1: argmin = (j,keys[0]-1,None)
            for i in range(len(keys)-1):
                ids = inds[keys[i]]
                for idx in ids:
                    id2.remove(idx)
                    id1.add(idx)
                res = len(id1)/len(indices) * gini_inpurity(self.X,self.y,list(id1),self.num_class) +\
                       len(id2)/len(indices) * gini_inpurity(self.X,self.y,list(id2),self.num_class)
                if res < MIN:
                    MIN = res
                    argmin = (j,(keys[i]+keys[i+1])/2, keys[i])

        inds = {}
        j = argmin[0]
        for i in indices:
            x = self.X[i,j]
            if x in inds.keys(): inds[x].append(i)
            else: inds[x] = [i]
        keys = sorted(inds.keys())
        id1,id2 = set(),set(indices)
        for i in range(len(keys)-1):
            if argmin[-1] == None: break
            ids = inds[keys[i]]
            for idx in ids:
                id2.remove(idx)
                id1.add(idx)
            if keys[i] == argmin[-1]: break

        return MIN, argmin[:-1], list(id1), list(id2)