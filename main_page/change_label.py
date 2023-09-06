import numpy as np
import os

def label_to_str(cell_label_number):
    labels = {0: 'rbc', 1:'wbc',2:'plt',3:'agg',4:'oof'}
    label_str = labels[cell_label_number]
    return  label_str


def cell_label_update(file_path, index, new_label):
     data = np.load(file_path, allow_pickle = True)
     if index in data.item():
          data.item()[index]['label'] = new_label
          np.save('updated_data.npy',data)

          # Check if the label has been changed successfully
          data_updated = np.load('updated_data.npy', allow_pickle=True)
          if index in data_updated.item() and data_updated.item()[index]['label'] == new_label:
            return 'Label has been changed successfully.'
          else:
            return 'Failed to change the label.'

          
def updated_labels(file_path_new_data, index, new_label):
    # Check if the file exists
    if os.path.isfile(file_path_new_data):
        # Load existing data
        existing_data = np.load(file_path_new_data, allow_pickle=True).item()
    else:
        # Initialize an empty list if the file does not exist
        existing_data = {}

    # Create a new dictionary with the given index and label
    new_data = { 'idx': index,
                 'label': new_label
                 }
    
    #check if cell has been changed in the same active learning turn
    #existing_entry = next((entry for entry in existing_data if entry['idx'] == index), None)

    existing_entry  = None
    if len(existing_data) != 0:
        exit_ind = [entry['idx'] for entry in existing_data.values()]
        exit_label = [entry['label'] for entry in existing_data.values()]
        for i in range(len(exit_ind)):
            if exit_ind[i] == index and exit_label[i] == new_label:
                existing_entry  = existing_data[i]
                break


    if existing_entry is None:
        # Add new data to existing data if index does not exist
        new_key = len(existing_data)
        existing_data[new_key] = new_data
    else:
        # Update the label of the existing data if index exists
        existing_entry['label'] = new_label

    # Save updated data back to file
    #existing_data = convert_list_to_dict(existing_data)
    np.save(file_path_new_data, existing_data)

    #check if the label has been updated successfully
    data_test = np.load(file_path_new_data, allow_pickle=True).item()
    #[data_dict] =data_test.values()
    # Get the last entry
    #[data_dict] =data_test.values()
    data_ind = [entry['idx'] for entry in data_test.values()]
    data_label = [entry['label'] for entry in data_test.values()]
    updated_entry = None
    for i in range(len(data_ind)):
        if data_ind[i] == index and data_label[i] == new_label:
            updated_entry = data_test[i]
            break
        


    #updated_entry = next(([entry] for entry in data_test.values()if entry['idx'] == index), None)

    # Check if the index and label match
    if updated_entry is not None :
        return 'Update successful!'
    else:
        return 'Failed to change the label.'

    
    