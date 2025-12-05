#!/bin/bash

# Download files using wget
download_files() {
    wget -q --show-progress "https://lobogit.unm.edu/jschueler1/oasis_sample_data/-/blob/main/data/dataframes/NR.pkl"
    wget -q --show-progress "https://lobogit.unm.edu/jschueler1/oasis_sample_data/-/blob/main/data/dataframes/hybrid.pkl"
    wget -q --show-progress "https://lobogit.unm.edu/jschueler1/oasis_sample_data/-/blob/main/data/models/no_weights.pt"
    wget -q --show-progress "https://lobogit.unm.edu/jschueler1/oasis_sample_data/-/blob/main/data/models/nominal_weights.pt"
    wget -q --show-progress "https://lobogit.unm.edu/jschueler1/oasis_sample_data/-/blob/main/data/models/overlap_only_weights.pt"
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
    echo "Downloading zip files"
    download_files
    echo "Done!"
    echo "Making relevant directories"
    makedirs

    echo "Moving files to their appropriate directories"
    mv NR.pkl ../data/dataframes
    mv hybrid.pkl ../data/dataframes
    mv no_weights.pt ../data/models
    mv nominal_weights.pt ../data/models
    mv overlap_only_weights.pt ../data/models

    echo "SUCCESS!"
}

# Execute main function
main
