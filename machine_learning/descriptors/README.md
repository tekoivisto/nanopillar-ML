Codes for generating the input descriptors for the CNN, which are grain boundary atom density, lattice orientations, combination of grain boundary atom densities and lattice orientations, and atom positions.

Prior to running descriptors_h5.py and descriptors.py, lattice orientations and structure types should be computed using Ovito's polyhedral template matching algorithm. This is done by following these steps:
1. Download the nanopillars from Zenodo
2. Run molecular_dynamics/nanopillars/npy_to_lmp.py to convert the nanopillars to Ovito readable format.
3. Open the produced .lmp file with Ovito
4. In Ovito, select Add modification -> Polyhedral template matching.
5. Select File -> Export file. In the opened window select folder "raw" inside this folder. Write File name: descriptors_test_set or descriptors_tr_val_set depending on which set you are working on. Select Save as type: XYZ File (*). Click Save.
6. In the opened Data Export Settings, select Range and File sequence. Click OK to export the data.
7. Do steps 2-6 for both test_set and tr_val_set. In npy_to_lmp.py, the set is changed by changing the commented line at the top of the script.
