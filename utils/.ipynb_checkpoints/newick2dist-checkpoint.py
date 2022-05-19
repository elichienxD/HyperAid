import numpy as np

def get_all_leaves(tree):
    if tree.name != None:
        # Already a leaves
        return [int(tree.name)]
    else:
        Left = get_all_leaves(tree.descendants[0])
        Right = get_all_leaves(tree.descendants[1])
        return Left+Right

def newick2dist(trees,n):
    # This convert a tree from newick format to distance matrix (n-by-n)
    # Note that all internal nodes are named "None"
    # All leaves node have name in np.arange(n)
    
    D = np.zeros((n,n))
    assert len(trees[0].descendants) == 2 # An internal/root node should always have 2 children!
    
    
    D = Fill_D_from_tree(trees[0],D)
    return D

def Fill_D_from_tree(subtree,D):
    assert len(subtree.descendants) == 2 # An internal nodes should always have 2 children!
    All_leaves = np.arange(D.shape[0])    
    
    Left = subtree.descendants[0]
    Left_leaves = get_all_leaves(Left)
    Right = subtree.descendants[1]
    Right_leaves = get_all_leaves(Right)
    
    D[np.ix_(Left_leaves,np.setdiff1d(All_leaves,Left_leaves))] += Left.length
    D[np.ix_(np.setdiff1d(All_leaves,Left_leaves),Left_leaves)] += Left.length
    D[np.ix_(Right_leaves,np.setdiff1d(All_leaves,Right_leaves))] += Right.length
    D[np.ix_(np.setdiff1d(All_leaves,Right_leaves),Right_leaves)] += Right.length
    
    if Left.name == None:
        D = Fill_D_from_tree(Left,D)
    if Right.name == None:
        D = Fill_D_from_tree(Right,D)
    
    return D

def save_dist2phylip(D,path):
    n = D.shape[0]
    f = open(path, "w")
    f.write(str(n)+'\n')
    for i in range(n):
        f.write(str(i)+' '+str(D[i])[1:-1]+'\n')

    f.close()
    print("Matrix saved")
    return 