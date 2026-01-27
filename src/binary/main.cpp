#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "inf_binary.h"    // inference declarations
#include "io_binary.h"     // input/output

/*********************************************************************
 
                    COMMAND LINE INPUT FORMAT
 
 Command line instructions tell the program where to look for 
 input files and where to send output, as well as the setting of 
 parameters. Numerical parameters may be entered in either 
 scientific (recommended) or standard decimal notation. True/false 
 switches are off by default - if entered in the command line, the 
 corresponding option is set to true. Default parameter values can
 also be found in inf.h.
 
 The conventions are given below:
 
 -(flag name): (type of input expected)
 
 -d: string
    Default: "." (current directory)
    Path to the directory where the data file is located, and 
    where output will be written.
 
 -i: string
    Default: "input.dat"
    The location of the file containing the input observations.
    Each observation consists of a time (i.e. generation) that a 
    sequence is observed, the number of such identical sequences 
    observed, and the value of all tracked alleles (integer from
    0 to q-1).Each row represents one observation, and each 
    column is one variable:
        t1  n1  0   0   ...
        t1  n2  1   0   ...
        t2  n3  0   0   ...
    
 -o: string
    Default: "output.dat"
    The location of the file where output is to be sent.
    
 -m: string
    Default: "mu.dat"
    The location of the file containing the mutation matrix, in
    the case of a model with multiple states. If used, then the
    number of states q will be set by the dimensions of the matrix.

 -e: string
    Default: "po.dat"
    The location of the file containing the trait site information, 
    in the case of a model with multiple states.

 -es: string
    Default: "poSequence.dat"
    The location of the file containing the trait sequence 
    information, in the case of a model with multiple states.

 -ed: string
    Default: "poDis.dat"
    The location of the file containing the distance between
    two trait sites.

 -t: real number
    Default: 0.05
    Largest allowed absolute value of difference between covariance
    matrices before interpolation is triggered.

 -g: real number
    Default: 1
    Gaussian regularization strength for selection coefficients.
    
 -N: real number
    Default: 1.0e4
    Population size.
 
 -mu: real number
    Default: 2.0e-4
    Mutation rate per generation.

 -rr: real number
    Default: 2.0e-4
    recombination rate per generation.
    
 -a: none
    Enable "asymptotic" inference. We assume that the input sequences
    are collected over a very long time, and generations are ignored.
    In this case only the mutation term contributes in the numerator.
 
 -v: none
    Enable verbose output.

 *********************************************************************/


// MAIN PROGRAM

int main(int argc, char *argv[]) {
    
    // PROCESS COMMAND LINE INPUT
    
    RunParameters(r);
    
    for (int i=1;i<argc;i++) {
        
        // Location of input/output files
        if      (strcmp(argv[i],"-d")==0)  { if (++i==argc) break; else r.directory  = argv[i];                     }// directory
        else if (strcmp(argv[i],"-i")==0)  { if (++i==argc) break; else r.infiles.push_back(argv[i]);               }// input file name
        else if (strcmp(argv[i],"-o")==0)  { if (++i==argc) break; else r.outfile    = argv[i];                     }// output file name
        else if (strcmp(argv[i],"-m")==0)  { if (++i==argc) break; else { r.muInfile = argv[i]; r.useMatrix=true; } }// mutation matrix
        else if (strcmp(argv[i],"-e")==0)  { if (++i==argc) break; else r.traitInfile = argv[i];                    }// trait sites
        else if (strcmp(argv[i],"-es")==0) { if (++i==argc) break; else r.traitSequence = argv[i];                  }// WT sequences for trait groups
        else if (strcmp(argv[i],"-ed")==0) { if (++i==argc) break; else r.traitDis = argv[i];                       }// distance between 2 trait sites

        // Regularization strength and parameter settings
        else if (strcmp(argv[i],"-t")==0)  { if (++i==argc) break; else r.tol   = strtodouble(argv[i]);             }
        else if (strcmp(argv[i],"-g")==0)  { if (++i==argc) break; else r.gamma = strtodouble(argv[i]);             }// regularization force
        else if (strcmp(argv[i],"-N")==0)  { if (++i==argc) break; else r.N     = strtodouble(argv[i]);             }// population number
        else if (strcmp(argv[i],"-mu")==0) { if (++i==argc) break; else r.mu    = strtodouble(argv[i]);             }// mutation rate
        else if (strcmp(argv[i],"-rr")==0) { if (++i==argc) break; else r.rr    = strtodouble(argv[i]);             }// recombination rate
        
        // Optional output/processing
        else if (strcmp(argv[i],"-nc")==0) { r.useCovariance  = false;                                                       }
        else if (strcmp(argv[i],"-sc")==0) { if (++i==argc) break; else { r.covOutfile = argv[i]; r.saveCovariance = true; } }
        else if (strcmp(argv[i],"-sn")==0) { if (++i==argc) break; else { r.numOutfile = argv[i]; r.saveNumerator  = true; } }
        else if (strcmp(argv[i],"-a")==0)  { r.useAsymptotic  = true;                                                        }
        else if (strcmp(argv[i],"-v")==0)  { r.useVerbose     = true;                                                        }
        
        else printf("Unrecognized command! '%s'\n",argv[i]);
                
    }
    
    return run(r);
    
}
