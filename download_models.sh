#!/bin/bash

# Clone the repository
git clone https://github.com/cmu-argus-1/NN-models

# Move the 'rc' and 'ld' folders into the 'models/' folder
mkdir -p models
mv NN-models/rc models/
mv NN-models/ld models/

# Remove the cloned repository
rm -rf NN-models

echo "The rc and ld networks have been successfully moved to the models/ directory."
