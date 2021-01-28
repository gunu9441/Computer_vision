def sumVector(*args : tuple) -> tuple :
    #args = ((1,2),(2,3),(3,4))
    print(args)
    output_list = [sum(i) for i in zip(*args)]
    print(output_list)
    output_tuple = tuple(output_list)
    print(output_tuple)



vectors = [(1,2),(2,3),(3,4)]
sumVector(*vectors)