#!/bin/bash

# To return to the initial state, first run the "back2beginning.sh" script.
echo "Delete the intermediate generated files and return to the initial state of the code."
./back2beginning.sh

# Data preprocessing.
echo "Perform data preprocessing."
config_file="./config/OPSA_config.yaml" 
sed -i "/^data:/,/^[^ ]/ s/\(train_path:\).*/\1 '.\/train_tisv'/" "$config_file" 
python data_preprocess.py 

# Search for triggers.
sed -i 's/training: !!bool "false"/training: !!bool "true"/' "$config_file"
if [ -f "trigger_token_ids.npy" ]; then
		echo "Trigger search completed."
	else
		echo "Search for triggers."
		python search_trigger.py
		echo "Trigger search completed."
fi

# Data poisoning.
echo "Data poisoning."
sed -i 's/training: !!bool "false"/training: !!bool "true"/' "$config_file" 
sed -i '/^poison:/,/^[^ ]/ s/\(p_people:\).*/\1 !!float "0.01"/' "$config_file"
sed -i '/^poison:/,/^[^ ]/ s/\(p_utte:\).*/\1 !!float "0.99"/' "$config_file"
python data_poison.py

# Model training.
echo "Train the model! Relevant information will be saved in the log."
sed -i "/^data:/,/^[^ ]/ s/\(train_path:\).*/\1 '.\/train_tisv_poison_cluster'/" "$config_file" 
python train_poison_speech_embedder.py 

# Perform clean performance testing of the model.
echo "Test the model for normal performance! Relevant information will be saved in the log."
sed -i 's/training: !!bool "true"/training: !!bool "false"/' "$config_file" 
python train_poison_speech_embedder.py 

# Perform adversarial performance testing.
echo "Test the effectiveness of attacks! Relevant information will be saved in the log."
python sh_utils.py
temp_file="./temp"  
a=$(cat "$temp_file")
sed -i '/^poison:/,/^[^ ]/ s/\(threash:\).*/\1 !!float "'"$a"'"/' "$config_file"
rm -rf "$temp_file"
python test_speech_embedder_poison.py 
