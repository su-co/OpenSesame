# Delete the files created by the intermediate process and return to the original state

dir=$(pwd)


find "$dir" -type d -name "test_tisv*" -exec rm -rf {} \;
echo "The directories starting with 'test_tisv' have been deleted."


find "$dir" -type d -name "train_tisv*" -exec rm -rf {} \;
echo "The directories starting with 'train_tisv' have been deleted."


find "$dir" -type d -name "poison_speaker_cluster*" -exec rm -rf {} \;
echo "The directories starting with 'poison_speaker_cluster' have been deleted."

rm -rf "$dir/trigger_base"
echo "The directories starting with 'trigger_base' have been deleted."
rm -rf "$dir/validate_tisv"
echo "The directories starting with 'validate_tisv' have been deleted."


#find "$dir" -type d -name "*.npy" -exec rm -rf {} \;
#echo "The directories ending with '.npy' have been deleted."
