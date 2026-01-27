#ifndef INF_H
#define INF_H

#include <vector>
#include <string>
#include <stdio.h>


// Typedefs
typedef std::vector<std::vector<double> > Vector;
typedef std::vector<std::vector<int> > IntVector;
typedef std::vector<std::vector<std::vector<double> > > VVector;
typedef std::vector<std::vector<std::vector<int> > > IntVVector;


// PROGRAM SETTINGS - This class holds the parameters needed for running the algorithm

class RunParameters {
    
public:
    
    std::string directory;             // path to the directory where the inut file is located
                                       // output will also be sent to this directory
    std::vector<std::string> infiles;  // input file list
    std::string muInfile;              // input file for mutation matrix
    std::string traitInfile;           // input file for trait site information
    std::string traitSequence;         // input file for trait sequence information
    std::string traitDis;              // input file for distance tween 2 trait sites
    std::string outfile;               // output file
    std::string covOutfile;            // output file for the regularized integrated covariance matrix
    std::string numOutfile;            // output file for the "numerator" (change in mutant frequency + mutation term)
    
    double tol;             // maximum tolerance for covariance differences before interpolating
    double gamma;           // Gaussian regularization strength
    double N;               // population size
    double mu;              // mutation rate per generation
    double rr;              // recombination rate per generation
    
    bool useMatrix;         // if true, read mutation matrix from file
    bool useCovariance;     // if true, include covariance (linkage) information, else default to independent sites
    bool useAsymptotic;     // if true, assume that sequences are collected over long times (equilibrium)
    bool useVerbose;        // if true, print extra information while program is running
    bool saveCovariance;    // if true, output the total covariance matrix
    bool saveNumerator;     // if true, output the "numerator" multiplying the inverse covariance
    
    RunParameters() {
        
        directory     = ".";
        muInfile      = "mu.dat";
        outfile       = "output.dat";
        traitInfile   = "po.dat";
        traitSequence = "poS.dat";
        traitDis      = "poDis.dat";
        
        tol    = 0.05;
        gamma  = 1.0e3;
        N      = 1.0e3;
        mu     = 2.0e-4;
        rr     = 2.0e-4;
        
        useMatrix      = false;
        useCovariance  = true;
        useAsymptotic  = false;
        useVerbose     = false;
        saveCovariance = false;
        saveNumerator  = false;
        
    }
    std::string getSequenceInfile()              { return (directory+"/"+infiles[0]);   }
    std::string getSequenceInfile(int i)         { return (directory+"/"+infiles[i]);   }
    std::string getMuInfile()                    { return (directory+"/"+muInfile);     }
    std::string getTraitInfile()                 { return (directory+"/"+traitInfile);  }
    std::string getTraitSInfile()                { return (directory+"/"+traitSequence);}
    std::string getTraitDInfile()                { return (directory+"/"+traitDis);     }
    std::string getSelectionCoefficientOutfile() { return (directory+"/"+outfile);      }
    std::string getCovarianceOutfile()           { return (directory+"/"+covOutfile);   }
    std::string getNumeratorOutfile()            { return (directory+"/"+numOutfile);   }
    ~RunParameters() {}
    
};

// Main program
int run(RunParameters &r);

// Auxiliary routines
void computeAlleleFrequencies(const IntVector &sequences, const std::vector<double> &counts, const IntVector &trait_sites,const IntVector &trait_sequence, std::vector<double> &p1, std::vector<double> &p2,std::vector<double> &pt);
void computeRecFrequencies(const IntVector &sequences, const std::vector<double> &counts, const IntVector &trait_sites,const IntVector &trait_sequence, std::vector<double> &pk);
void updateCovarianceIntegrate(double dg, const std::vector<double> &p1_0,const std::vector<double> &p2_0, const std::vector<double> &p1_1, const std::vector<double> &p2_1,double totalCov[]); 
void updateMuIntegrate(double dg, int L, const Vector &muMatrix, const IntVector &trait_sites, const IntVector &trait_sequence, const std::vector<double> &p1_0, const std::vector<double> &pt_0, const std::vector<double> &p1_1, const std::vector<double> &pt_1, std::vector<double> &totalMu);
void updateComIntegrate(double dg, int L, double rec, const IntVector &trait_sites, const IntVector &trait_sequence, const IntVector &trait_dis, const std::vector<double> &p1_0, const std::vector<double> &pk_0, const std::vector<double> &p1_1, const std::vector<double> &pk_1, std::vector<double> &totalCom);
void processStandard(const IntVVector &sequences, const Vector &counts, const std::vector<double> &times, const Vector &muMatrix, const IntVector &trait_sites,const IntVector &trait_sequence,const IntVector &trait_dis, double r_rate, double totalCov[], double dx[]);
void regularizeCovariance(const IntVVector &sequences, int ne, double gammaN, double totalCov[]);

#endif
