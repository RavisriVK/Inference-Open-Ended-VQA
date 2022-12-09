import csv

# open the file in the write mode
with open('captioning_samples.tsv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f, delimiter='\t', lineterminator='\n')

    writer.writerow(['images/airplane.jpg'])
    writer.writerow(['images/donuts.png'])
    writer.writerow(['images/house-party.png'])
    writer.writerow(['images/panda-tree.png'])
    writer.writerow(['images/pizza-girl.png'])

with open('answering_samples.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t', lineterminator='\n')

    writer.writerow(['images/airplane.jpg', 'What is this?'])
    writer.writerow(['images/donuts.png', 'What is under the donuts'])
    writer.writerow(['images/house-party.png', 'Are they drinking wine?'])
    writer.writerow(['images/panda-tree.png', 'Is the panda alone?'])
    writer.writerow(['images/pizza-girl.png', 'Does the child have a plate?'])