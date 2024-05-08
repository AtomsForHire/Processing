# Processing

Python program to get various metrics of hyperdrive calibration solutions and images.

Requires: https://github.com/Chuneeta/mwa_qa

requires data.in file formatted like this:

working dir
save rms dir
save cal dir
save smoothness dir
dir to calibration metafits
<cyclic/sorted> <period>
<pointing centers>

Example:

./
./rms_cal/both/
./var_cal/both/
./smoothness/
../rerun_1_solutions/
cyclic 3
pc 1
pc 2
pc 3

"Cyclic" means the observations can be grouped in terms of different pointing centres.
For example, observation 1 can be pointing at pc 1, observation 2 can be pointing at pc 2,
ovservation 3 can be pointing at pc 3, and observation 4 points back to pc 1 etc. Here the 
period of pointing centres is 3. This results in some figures having distinct sections in 
the graph, for example the dynamic range plot may have 3 distinct lines where the first line
are observations pointing at pc 1, the second corresponding to pc 2 and the third, pc 3.
Looking at the x-axis as a whole, one would see the observations are not strictly increasing
but just looking at a particular line, the ids would be strictly increasing.

"Sorted" just means we do everything in order of obsid.
