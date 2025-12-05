#!/bin/bash

# Download files using wget
download_files() {
    wget -q --show-progress "https://lobogit.unm.edu/jschueler1/oasis_sample_data/-/raw/main/data/dataframes/NR.pkl?ref_type=heads"
    wget -q --show-progress "https://lobogit.unm.edu/jschueler1/oasis_sample_data/-/raw/main/data/dataframes/hybrid.pkl?ref_type=heads"
    wget -q --show-progress "https://lobogit.unm.edu/jschueler1/oasis_sample_data/-/raw/main/data/models/no_weights.pt?ref_type=heads"
    wget -q --show-progress "https://lobogit.unm.edu/jschueler1/oasis_sample_data/-/raw/main/data/models/nominal_weights.pt?ref_type=heads"
    wget -q --show-progress "https://lobogit.unm.edu/jschueler1/oasis_sample_data/-/raw/main/data/models/overlap_only_weights.pt?ref_type=heads"
}

# Function to make directories
makedirs() {
    local paths=("../data/"
		 "../data/dataframes"
                 "../data/models")
    for path in "${paths[@]}"; do
        if [ ! -d "$path" ]; then
            mkdir -p "$path"
        fi
    done
}

# Main function
main() {
    echo "Downloading data and weights files"
    download_files
    echo "Done!"
    echo "Making relevant directories"
    makedirs

    echo "Moving files to their appropriate directories"
    mv "NR.pkl?ref_type=heads" ../data/dataframes/NR.pkl
    mv "hybrid.pkl?ref_type=heads" ../data/dataframes/hybrid.pkl
    mv "no_weights.pt?ref_type=heads" ../data/models/no_weights.pt
    mv "nominal_weights.pt?ref_type=heads" ../data/models/nominal_weights.pt
    mv "overlap_only_weights.pt?ref_type=heads" ../data/models/overlap_only_weights.pt

    echo "SUCCESS!"
}

# Execute main function
main
