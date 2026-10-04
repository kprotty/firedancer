A='123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
def b58(h):
    n=int(h,16); s=''
    while n: n,r=divmod(n,58); s=A[r]+s
    z=0
    for i in range(0,len(h),2):
        if h[i:i+2]=='00': z+=1
        else: break
    return '1'*z+s
