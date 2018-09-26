py test.py --phase test --policy cadrl --model_dir data/final_circle_5p/cadrl_circle_5p_visible/ --visualize --test_case $N --traj &
py test.py --phase test --policy lstm_rl --model_dir data/final_circle_5p/lstm_rl_circle_5p_visible/ --visualize --test_case $N --traj &
py test.py --phase test --policy sarl --model_dir data/final_circle_5p/sarl_circle_5p_visible/ --visualize --test_case $N --traj &
py test.py --phase test --policy sarl --model_dir data/final_circle_5p/om-sarl_circle_5p_visible/ --visualize --test_case $N --traj &
