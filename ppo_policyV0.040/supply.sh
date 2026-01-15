run_time=24.0
test_num=100

env_name="N-ex"
part_num=25
dist_type='h'

for N in 9;  
do  
  #python main.py --resblock-num $N --env-name $env_name --part-num $part_num  --dist-type $dist_type --run-hours $run_time --num-processes 16 --num-steps 512 --num-mini-batch 32 --log-interval 1 --eval-interval 100 --eval-num 80 --excel-save ;
  python enjoy.py --resblock-num $N --env-name $env_name --part-num $part_num --dist-type $dist_type --not-eval-load --num-processes 10; 
done 



