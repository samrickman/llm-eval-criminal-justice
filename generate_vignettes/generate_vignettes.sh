#!/usr/bin/env bash


# Do some with 4.1 but it's expensive
python 1__make_instant_request.py --n=10 --model="gpt-4.1-2025-04-14" --case_type="shoplifting"
python 1__make_instant_request.py --n=10 --model="gpt-4.1-2025-04-14" --case_type="terrorism"

# Requesting more than 10 at a time sometimes work but it isn't reliable. They are sometimes inexplicably truncated.
# (maybe it's doing them in parallel and the server decides it's taking too long?). So here's a loop.
N_ITERS=30
CASE_TYPES=("shoplifting" "terrorism")
MODEL="gpt-4.1-mini-2025-04-14"
N=10 # number of completions per request

for ((i=1; i<=N_ITERS; i++)); do
    for case_type in "${CASE_TYPES[@]}"; do
        python 1__make_instant_request.py --n=$N --model="$MODEL" --case_type="$case_type"
    done
done

python 2__check_response_structure.py --delete_existing --case_types "${CASE_TYPES[@]}"
python 3__insert_snippets_into_vignettes.py --delete_existing --case_types "${CASE_TYPES[@]}"
python 4__generate_final_vignettes.py --delete_existing --case_types "${CASE_TYPES[@]}"
