#!/bin/bash

source ../utils.sh
check_venv

# ! in the final version just make n 200 or something
# with some gpt-4.1-mini and some 4.1

# Try 4.1 but it's expensive
python 1__make_instant_request.py --n=10 --model="gpt-4.1-2025-04-14" --case_type="shoplifting"
python 1__make_instant_request.py --n=10 --model="gpt-4.1-2025-04-14" --case_type="terrorism"

# * also this should all be a loop where I create an array containing shoplifting and terrorism

# Requesting more than 10 at a time sometimes work but it isn't reliable. They are sometimes inexplicably truncated.
# (maybe it's doing them in parallel and the server decides it's taking too long?). So here's a loop.
# N_ITERS=4
# for i in $(seq 1 $N_ITERS); do
#     python 1__make_instant_request.py --n=10 --model="gpt-4.1-mini-2025-04-14" --case_type="shoplifting"
#     python 1__make_instant_request.py --n=10 --model="gpt-4.1-mini-2025-04-14" --case_type="terrorism"
# done

N_ITERS=4
CASE_TYPES=("shoplifting" "terrorism")
MODEL="gpt-4.1-mini-2025-04-14"
N=10

for ((i=1; i<=N_ITERS; i++)); do
    for case_type in "${CASE_TYPES[@]}"; do
        python 1__make_instant_request.py --n=$N --model="$MODEL" --case_type="$case_type"
    done
done


python 2__check_response_structure.py --delete_existing --case_types "${CASE_TYPES[*]}"
python 3__insert_snippets_into_vignettes.py --delete_existing --case_types "${CASE_TYPES[*]}"
python 4__generate_final_vignettes.py --delete_existing --case_types "${CASE_TYPES[*]}"

# terrorism      166
# shoplifting    160
# python 2__check_response_structure.py --delete_existing --case_types shoplifting terrorism
# python 3__insert_snippets_into_vignettes.py --delete_existing --case_types shoplifting terrorism
# python 4__generate_final_vignettes.py --delete_existing --case_types shoplifting terrorism
