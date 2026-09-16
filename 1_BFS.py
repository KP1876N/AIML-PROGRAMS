from collections import deque
g={0:[1,3],1:[4,2],2:[5],3:[2],4:[5],5:[]}
def bfs(s):
    q=deque([s]); v={s}
    while q:
        n=q.popleft()
        print(n,end=" ")
        for x in g[n]:
            if x not in v:v.add(x);q.append(x)

def dfs(n,v=None):
    v=set();v.add(n);print(n,end=" ")
    for x in g[n]:
        if x not in v:dfs(x,v)
print("Following is Breadth First Traversal (starting from vertex 3):")
bfs(3)
print("\nFollowing is DFS from (starting from vertex 2)")
dfs(2)
