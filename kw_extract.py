from keybert import KeyBERT

# Step 1: Instantiate the KeyBERT model
model = KeyBERT('distilbert-base-nli-mean-tokens', min_ngram=1, max_ngram=2)

# Step 2: Extract keywords for each dialogue in the dataset
keywords_dataset = []

for dialogue, label in dataset:
    keywords = model.extract_keywords(dialogue, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    keywords_dataset.append((dialogue, label, keywords))

# Step 3: Calculate the frequency of each keyword that appears in a specific label
label = "Example Label"
keyword_frequency = {}

for dialogue, label, keywords in keywords_dataset:
    # Iterate over each keyword in the current dialogue
    for keyword in keywords:
        # Check if the keyword is already in the dictionary
        if keyword in keyword_frequency:
            # If it is, increment the frequency count
            keyword_frequency[keyword] += 1
        else:
            # If it's not, add it to the dictionary with a frequency of 1
            keyword_frequency[keyword] = 1

label_keyword_frequency = {}

for keyword, frequency in keyword_frequency.items():
    # Check if the current keyword appears in the desired label
    if label in [l for _, l, _ in keywords_dataset if l == label]:
        # If it does, add it to the label keyword frequency dictionary
        label_keyword_frequency[keyword] = frequency

# Step 4: Sort the label keyword frequency dictionary in descending order of frequency
sorted_label_keyword_frequency = dict(sorted(label_keyword_frequency.items(), key=lambda item: item[1], reverse=True))

# Step 5: Print or save the results
for keyword, frequency in sorted_label_keyword_frequency.items():
    print(f'{keyword}: {frequency}')
