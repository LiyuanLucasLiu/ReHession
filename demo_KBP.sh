unzip Data/intermediate/KBP.zip -d Data/intermediate/
python2 DataProcessor/relation_feature_generation.py
cd Model
make
cd ..
./Model/ReHession -train ./Data/intermediate/KBP/train.data -test ./Data/intermediate/KBP/test.data -none_idx 6 -instances 225977 -test_instances 2111
#./Model/ReHession -train ./Data/intermediate/NYT/train.data -test ./Data/intermediate/NYT/test.data -none_idx 0 -instances 530767 -test_instances 3803
