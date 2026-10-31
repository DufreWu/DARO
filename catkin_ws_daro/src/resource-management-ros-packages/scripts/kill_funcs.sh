######################################
function kill_all_screens() {
    for session in $(screen -ls | grep -o '[0-9]*\.chen_[[:alnum:]_]*' | grep -Ev "chen_core|chen_davis_driver|run"); do  #  
        echo "Killing screen session: $session"
        screen -S "${session}" -X quit;
    done
}

function kill_edge_image() {
    rosnode kill "edges";
    sleep 1;
    screen -S "chen_edge_image" -X quit; 
    echo "edges killed!"
}

function kill_denoising_filling() {
    rosnode kill "denoising_filling";
    sleep 1;
    screen -S "chen_denoising_filling" -X quit; 
    echo "denoising_filling killed!"
}

function kill_distance_surface() {
    rosnode kill "distance_surface";
    sleep 1;
    screen -S "chen_distance_surface" -X quit; 
    echo "distance_surface killed!"
}

function kill_optical_flow() {
    rosnode kill "optical_flow";
    sleep 1;
    screen -S "chen_optical_flow" -X quit; 
    echo "chen_optical_flow killed!"
}

function kill_tracker() {
    rosnode kill "tracker";
    sleep 1;
    screen -S "chen_tracker" -X quit; 
    echo "chen_tracker killed!"
}