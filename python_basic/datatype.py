# list
print('-------list-------')

mylist = []
mylist.append(1)
mylist.append(2)
mylist.append(3)
mylist.append(1)
mylist.append(2)
mylist.append(3)
#1,2,3,1,2,3
print('list: {}'.format(mylist))

# 앞에 있는 1부터 삭제
mylist.remove(1)
print('remove 1 using remove func')
print('list: {}'.format(mylist))

print('length: {}'.format(len(mylist)))

print('------------------')
print('-------list-------')
mylist = [1,2,3,4]
print('list: {}'.format(mylist))
print('mylist[0] : {}'.format(mylist[0]))
# 1

print('mylist[-1]: {}'.format(mylist[-1]))
# 4

print('mylist[-4]: {}'.format(mylist[-4]))
# 1

# print(mylist[-5])
# error

#Tuple
print('-------tuple-------')
mytuple = (1, 2, 3)
print('tuple: {}'.format(mytuple))
print('mytuple[1]: {}'.format(mytuple[1]))
print('------------------')

#set
print('--------set--------')
myset = set()
myset.add(1)
myset.add(2)
myset.add(3)
print('set: {}'.format(myset))

myset.remove(2)
print('remove 2 using remove func')
print('set: {}'.format(myset))
print('--------------------')

#dict
print('--------dict--------')
mydict = dict()
mydict['apple'] = 123
print('add key: \'apple\' value: 123')
print('dict: {}'.format(mydict))
print()
mydict[0] = 1
print('add key: 0 value: 1')
print('dict: {}'.format(mydict))
print()

mydict[3.14] = 1
print('add key: 3.14 value: 1')
print('dict: {}'.format(mydict))
print()

print('change value in \'apple\': 123 --> \'hello apple\'')
mydict['apple'] = 'hello apple'
print('dict: {}'.format(mydict))
print()

print('length: {}'.format(len(mydict)))

print('--------------------')
print('dict: {}'.format(mydict))

