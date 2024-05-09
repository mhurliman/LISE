SLDA static solver
Init MPI
create MPI groups - neutron (isospin=-1)/proton (isospin=1)

reads config file - broadcast parameters
sets broyden mixing and coulomb parameters
calculates various physical  properties

allocate density memory for proton/neutron
Generate the arrays of lattice values & kinetic energy array
Builds FFT plans for gradient calculations

Calculate density phases