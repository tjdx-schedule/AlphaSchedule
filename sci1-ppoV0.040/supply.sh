#conda activate pytorch_gpu
#python main.py --num-processes 16 --num-steps 512 --num-mini-batch 32 --log-interval 1 --eval-interval 100 --eval-num 80 --excel-save --env-name sci1-my --part-num 15 --run-hours 0.5
#python enjoy.py --env-name sci1-my --part-num 35

python enjoy.py --env-name sci1-my --part-num 95 --dist-type h --not-eval-load ; 

run_time=24.0
for i in 45;  
do  
  for j in m;
  do
    python main.py --num-processes 16 --num-steps 512 --num-mini-batch 32 --log-interval 1 --eval-interval 100 --eval-num 80 --excel-save --env-name sci1-my --part-num $i  --dist-type $j --run-hours $run_time;
    python enjoy.py --env-name sci1-my --part-num $i --dist-type $j --not-eval-load ; 
  done
done 
