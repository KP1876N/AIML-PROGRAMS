import heapq

G={
'A':[('B',2),('F',8)],'B':[('A',2),('C',4),('D',3)],
'C':[('B',4),('D',2),('E',3)],'D':[('B',3),('C',2),('E',4)],
'E':[('C',3),('D',4),('I',2),('J',7)],'F':[('A',8),('G',3),('H',4)],
'G':[('F',3),('I',5)],'H':[('F',4),('I',2)],
'I':[('E',2),('G',5),('H',2),('J',3)],'J':[]
}
H={'A':10,'B':8,'C':6,'D':7,'E':5,'F':8,'G':5,'H':4,'I':3,'J':0}

def astar(s,t):
    q=[(H[s],0,s,[s])]; seen={}
    while q:
        f,c,n,p=heapq.heappop(q)
        if n==t:
            print("Path found:",p)
            return
        if n in seen and seen[n]<=c:
            continue
        seen[n]=c
        for x,w in G[n]:
            heapq.heappush(q,(c+w+H[x],c+w,x,p+[x]))

astar('A','J')
