#!/bin/bash

trap "killall screen; exit 1;" INT;

if [ "$#" -ne 1 ]; then
    echo "run_experiment_const_all.sh <configs_file>"
fi

configs_file=$1
out_file=$configs_file.results

if [ -f $out_file ]; then
    echo "Result file exists, exit ..."
    exit 1;
fi

echo "index,speed,freq,mean_energy,std_energy,mean_th,std_th,min_th,max_th" > $out_file
cat $configs_file | while read line; do
    index=$(echo $line | cut -d "," -f 1);
    speed=$(echo $line | cut -d "," -f 2);
    freq=$(echo $line | cut -d "," -f 3);

    echo -e "Running config $index \t with speed: $speed \t and freq: $freq";

    exp_name="8.0-CC-$index-s$speed-f$freq";
    curr_time=`date +'%Y%m%d-%H%M%S'`;
    log_dir="./temp/exp-$exp_name-$curr_time"
    log_dir=$(realpath $log_dir);

    ./run_experiment_const_all_single.sh $log_dir $speed $freq;
    
    sleep 60;
    killall screen;

    echo 'adding the position and energy to the log, wait...'; 
    python3.9 utils/add_info_to_exp_log.py $log_dir; cat $log_dir/results_info.txt;
    
    echo 'plotting the experiment, wait...'; 
    python3.9 utils/plot_exp.py $log_dir;


    mean_energy=$(cat $log_dir/results_info.txt | grep -A 1 "ENERGY" | grep "mean" | cut -d ":" -f 2 | tr -d '\t ');
    std_energy=$(cat $log_dir/results_info.txt | grep -A 2 "ENERGY" | grep "std" | cut -d ":" -f 2 | tr -d '\t ');
    mean_th=$(cat $log_dir/results_info.txt | grep -A 4 "THROUGHPUT" | grep "mean" | cut -d ":" -f 2 | tr -d '\t ');
    min_th=$(cat $log_dir/results_info.txt | grep -A 4 "THROUGHPUT" | grep "min" | cut -d ":" -f 2 | tr -d '\t ');
    max_th=$(cat $log_dir/results_info.txt | grep -A 4 "THROUGHPUT" | grep "max" | cut -d ":" -f 2 | tr -d '\t ');
    std_th=$(cat $log_dir/results_info.txt | grep -A 4 "THROUGHPUT" | grep "std" | cut -d ":" -f 2 | tr -d '\t ');

    echo "$index,$speed,$freq,$mean_energy,$std_energy,$mean_th,$std_th,$min_th,$max_th";
    echo "$index,$speed,$freq,$mean_energy,$std_energy,$mean_th,$std_th,$min_th,$max_th" >> $out_file;

done;




