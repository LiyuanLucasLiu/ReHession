unzip Data/intermediate/NYT.zip -d Data/intermediate/
python2 DataProcessor/relation_feature_generation.py --Data NYT
cd Model
make
cd ..
./Model/ReHession -train ./Data/intermediate/NYT/train.data -test ./Data/intermediate/NYT/test.data -none_idx 0 -instances 530767 -test_instances 3803
