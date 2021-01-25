mylist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print('list: {}'.format(mylist))
print('even number in list:', end=' ')
for i in mylist:
    if i % 2 ==0:
        print(i, end=' ')

print()

# pythonic code
print([i for i in mylist if i % 2 == 0])