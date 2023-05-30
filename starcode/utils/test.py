def combine_lists(list1, list2): 
    new_list = [] 
    for i in range(len(list1)): 
        new_list.append(list1[i]) 
        new_list.append(list2[i]) 
    return new_list

print(combine_lists(['a', 'b', 'c'], [1, 2, 3])) 
# ['a', 1, 'b', 2, 'c', 3] print(combine_lists(['x', 'y', 'z'], [4, 5, 6])) # ['x', 4, 'y', 5, 'z', 6] print(combine_lists([1, 2, 3], [4, 5, 6])) # [1, 4, 2, 5, 3, 6]


number = int(input("Enter a number: ")) 
if number % 2 == 0: 
    print(f"{number} is an even number.") 
else: 
    print(f"{number} is an odd number.")