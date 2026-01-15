#conda activate pytorch_gpu
#python main.py --num-processes 16 --num-steps 512 --num-mini-batch 32 --log-interval 1 --eval-interval 100 --eval-num 80 --excel-save --env-name sci1-my --part-num 15 --run-hours 0.5
#python enjoy.py --env-name sci1-my --part-num 35
#rm -r /home/luopeng/桌面/alpha_ppo/compare/KNexperiment/N-ex-ppoV0.040/logs/excel/*
#rm -r /home/luopeng/桌面/alpha_ppo/compare/KNexperiment/N-ex-ppoV0.040/trained_models/ppo/*
#rm -r /home/luopeng/桌面/alpha_ppo/compare/KNexperiment/N-ex-K2/models/*
run_time=24.0
test_num=100

env_name="N-ex"
part_num=25
dist_type='h'

for N in 1 3 5 7 9 11 13 15;  
do  
  python main.py --resblock-num $N --env-name $env_name --part-num $part_num  --dist-type $dist_type --run-hours $run_time --num-processes 16 --num-steps 512 --num-mini-batch 32 --log-interval 1 --eval-interval 100 --eval-num 80 --excel-save ;
  python enjoy.py --resblock-num $N --env-name $env_name --part-num $part_num --dist-type $dist_type --not-eval-load --num-processes 10; 
done 

cd trained_models/ppo
cp *.model /home/luopeng/桌面/alpha_ppo/compare/KNexperiment/N-ex-K2/models
cd /home/luopeng/桌面/alpha_ppo/compare/KNexperiment/N-ex-K2

for N in 1 3 5 7 9 11 13 15;
do
  python testPolicy.py --resblock-num $N --mode K2 --env-name $env_name  --beam-size 10 --test-num $test_num
done


