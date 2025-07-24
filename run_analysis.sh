#!/usr/bin/env bash

# Define your array of environment variable names
REQUIRED_VARS=("OPENAI_API_KEY" "GEMINI_API_KEY" "ANTHROPIC_API_KEY", "GROK_API_KEY")

# Track unset variables
unset_vars=()

# Check each variable
for var_name in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var_name}" ]]; then
        unset_vars+=("$var_name")
    fi
done

# Output result
if [[ ${#unset_vars[@]} -eq 0 ]]; then
    echo "✅ All required environment variables are set."
    exit 0
else
    echo "❌ Error: The following environment variables are not set:"
    for var in "${unset_vars[@]}"; do
        echo "  - $var"
    done
    exit 1
fi

# * Generate vignettes
echo "Generating vignettes..."
cd generate_vignettes
chmod +x generate_vignettes.sh
./generate_vignettes.sh
echo "Vignettes generated."


# * Evaluate vignettes
echo "Evaluating vignettes..."
cd ../inspect_eval_vignettes
chmod +x evaluate.sh
./evaluate.sh
echo "Vignettes evaluated. Done!"
