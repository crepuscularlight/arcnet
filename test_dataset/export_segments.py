# Converting labelled dataset created from Test Data from Waste Detection Dataset using segments.ai

# Loading the Segments API
from segments import SegmentsClient
api_key = "cfc89b9c98b738859fab08d7f288b741ca170d17"
client = SegmentsClient(api_key)

user = "festay"
datasets = client.get_datasets(user)

# Listing available datasets
for dataset in datasets:
    print(dataset["name"], dataset["description"])

# Getting the desired dataset
dataset_identifier = user+"/arc_litter"
dataset = client.get_dataset(dataset_identifier)
print(dataset)

# Exprting to COCO format
# Export to COCO format
from segments.utils import export_dataset
#export_dataset(dataset, 'coco')


# # Alternatively, if obtained directly from a local file.
# pip install segments-ai --upgrade
from segments import SegmentsDataset
release_file = "/home/fidel/git/arcnet/test_dataset/arc_litter-v2.1.json"
dataset = SegmentsDataset(release_file, task='segmentation', filter_by=['labeled', 'reviewed'])
export_dataset(dataset, 'coco')

# The dataset is now saved into the RUNNING DIRECTORY

