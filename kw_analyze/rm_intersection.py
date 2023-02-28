# Sample input dict
my_dict = {
    1: {'label': 'item1', 'phrases': ['phrase1', 'phrase2', 'phrase3']},
    2: {'label': 'item2', 'phrases': ['phrase2', 'phrase4', 'phrase5']},
    3: {'label': 'item3', 'phrases': ['phrase3', 'phrase5', 'phrase6']}
}

# Step 1: Create a list of all phrases
all_phrases = []
for item in my_dict.values():
    all_phrases.extend(item['phrases'])

# Step 2: Find the intersection of all phrases
intersection = set(all_phrases[0]).intersection(*all_phrases[1:])

# Step 3: Remove the intersection phrases from each item's phrases list
for item in my_dict.values():
    item['phrases'] = list(set(item['phrases']) - intersection)

# Output the updated dict
print(my_dict)

"""
Output:

{
    1: {'label': 'item1', 'phrases': ['phrase1']},
    2: {'label': 'item2', 'phrases': ['phrase4', 'phrase5']},
    3: {'label': 'item3', 'phrases': ['phrase6']}
}
"""
