./Model/ReHession -train ./Data/intermediate/KBP/train.data -test ./Data/intermediate/KBP/test.data -threads 20 -none_idx 6 -cleng 150 -lleng 250 -negative 1 -resample 20 -ignore_none 0 -iter 25 -alpha 0.025 -debug 1 -dropout 0.3 -instances 225977 -test_instances 2111 -special_none 0 -error_log 1
./Model/ReHession -train ./Data/intermediate/NYT/train.data -test ./Data/intermediate/NYT/test.data -threads 20 -none_idx 0 -cleng 150 -lleng 250 -negative 1 -resample 20 -ignore_none 0 -iter 25 -alpha 0.025 -debug 1 -dropout 0.3 -instances 530767 -test_instances 3803 -special_none 0 -error_log 1


./Model/ReHession -train ./Data/intermediate/KBP/train.data -test ./Data/intermediate/KBP/test.data -threads 20 -none_idx 6 -cleng 50 -lleng 50 -negative 1 -resample 20 -ignore_none 0 -iter 25 -alpha 0.025 -debug 1 -dropout 0.0 -instances 225977 -test_instances 2111 -special_none 0 -error_log 1

./Model/ReHession -train ./Data/intermediate/NYT/train.data -test ./Data/intermediate/NYT/test.data -threads 20 -none_idx 0 -cleng 50 -lleng 50 -negative 1 -resample 20 -ignore_none 0 -iter 25 -alpha 0.025 -debug 1 -dropout 0.0 -instances 530767 -test_instances 3803 -special_none 0 -error_log 1