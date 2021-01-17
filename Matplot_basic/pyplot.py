import matplotlib.pyplot as plt

# plt.plot([1, 2, 3, 4])
# plt.show()

# plt.plot([1,2,3,4,5],[1,4,9,16,25])
# plt.plot([1,3,5],[1,9,25])
# plt.show()

# plt.plot([1,2,3,4,5],[1,4,9,16,25])
# plt.plot([1,3,5],[1,9,25])
# plt.xlabel('x label')
# plt.ylabel('y label')
# plt.show()

plt.figure(figsize=(5,5))
plt.plot([1,2,3,4,5], [1,4,9,16,25], 'bo', color = 'red', label = 'dot')
plt.plot([1,3,5],[1,9,25], label = 'line')
plt.xlabel('x label')
plt.ylabel('y label')
plt.axis([1,5,3,26])
# plt.xlim(1, 5)
# plt.ylim(3, 26)
plt.legend(loc='upper right')
plt.grid()
plt.show()
