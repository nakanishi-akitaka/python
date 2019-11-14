# def print_twice(word):
#     print(word)
#     print(word)
# 
# x=print_twice(2)
# x=print_twice(2*2)
# x=print_twice(2.0)
# x=print_twice(2.0*2)
# x=print_twice('2.0')
# x=print_twice('2.0'*2)
# 
# def print_70(word):
#     n=70-len(word)
#     print(' '*n+word)
# 
# print_70('word')
# print('1'*70)
# print_70('hoge hoge')
# print('1'*70)
# print_70('urtra soul')
# print('1'*70)
# def do_twice(f,x):
#     f(x)
#     f(x)
# 
# def do_four(f,x):
#     do_twice(f,x)
#     do_twice(f,x)
# 
# def print_twice(x):
#     print(x*2)
# 
# do_twice(print_twice,'test')
# do_four(print_twice,'test')

# print('+'+'-'*4+'+'+'-'*4+'+')
# print('|'+' '*4+'|'+' '*4+'|')
# print('|'+' '*4+'|'+' '*4+'|')
# print('|'+' '*4+'|'+' '*4+'|')
# print('|'+' '*4+'|'+' '*4+'|')
# print('+'+'-'*4+'+'+'-'*4+'+')
# print('|'+' '*4+'|'+' '*4+'|')
# print('|'+' '*4+'|'+' '*4+'|')
# print('|'+' '*4+'|'+' '*4+'|')
# print('|'+' '*4+'|'+' '*4+'|')
# print('+'+'-'*4+'+'+'-'*4+'+')
#
def print_abx(a,b,x):
    word=a
    for j in range(x):
        for i in range(4):
            word+=b
        word+=a
    print(word)

def print_lat(nl):
    print_abx('+','-',nl)
    for j in range(nl):
        for i in range(3):
            print_abx('|',' ',nl)
        print_abx('+','-',nl)

print_lat(2)
print_lat(3)
print_lat(4)
