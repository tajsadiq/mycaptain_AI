E = {0 ,2 ,4 ,6 ,8}
N = {1 ,2 ,3 ,4 ,5}

union = E.union(N)
inter = set.intersection(E,N)
diff = E-N
symdiff = union - inter

print('Union of E and N is ',union)
print('Intersection of E and N is ',inter)
print('Difference of E and N is',diff)
print('Symmetric difference of E and N is ',symdiff)
