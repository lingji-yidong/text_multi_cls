from collections import Counter

# list
my_list = [1, 2, 3, 1, 2, 1, 4]
my_counter = Counter(my_list)
print(my_counter)  # Counter({1: 3, 2: 2, 3: 1, 4: 1})

# tuple
my_tuple = ('a', 'b', 'c', 'a', 'b', 'a', 'd')
my_counter = Counter(my_tuple)
print(my_counter)  # Counter({'a': 3, 'b': 2, 'c': 1, 'd': 1})

# dictionary
my_dict = {'a': 2, 'b': 3, 'c': 1, 'd': 2}
my_counter = Counter(my_dict)
print(my_counter)  # Counter({'b': 3, 'a': 2, 'd': 2, 'c': 1})
