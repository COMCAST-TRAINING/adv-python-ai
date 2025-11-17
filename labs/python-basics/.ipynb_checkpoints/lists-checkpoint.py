fruits = ["apple", "banana", "cherry"]

# Access elements (indexing)
print(fruits[0])  # apple

# Modify elements
fruits[1] = "blueberry"
print(fruits)  # ['apple', 'blueberry', 'cherry']

# Append new item
fruits.append("mango")

# Remove item
fruits.remove("cherry")

# Loop through a list
for fruit in fruits:
    print(fruit)
