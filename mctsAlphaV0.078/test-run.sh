test_num=2

env_name="data"

for mode in "pure_policy";
do 
  for part in 15 25;
  do
    for dist_type in l m;
    do
      python testPolicy.py --mode $mode --part-num $part --env-name $env_name --test-num $test_num --dist-type $dist_type
    done
  done
done