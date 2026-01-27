# Copilot Instructions: Time-Varying Selection Coefficients

## Project Overview
This is a research codebase for inferring **time-varying selection coefficients** in HIV evolution and simulated populations. The project combines Python-based inference of evolutionary trajectories with C++ ODE solvers for efficiency.

**Key Paper**: "Inferring Fitness Seascapes from Evolutionary Histories" (Gao et al., bioRxiv 2025)

## Architecture

### Three-Layer Computation Stack

1. **Data Layer** (`data/`): Sequence alignments, mutation rates, and immunological data
   - `data/HIV/`: Patient-specific sequences and analysis results  
   - `data/simulation/`: Generated evolutionary trajectories (simple, trait, sigmoid models)

2. **Python Layer** (main analysis)
   - `inference_HIV.py`: ODE-based inference of time-varying selection for HIV patients
   - `simulation.py`: Population genetics simulations (Wright-Fisher model with mutations/recombination)
   - `epitope.py`: Sequence processing, alignment operations, trait definitions

3. **C++ Layer** (`src/`): Fast constant-selection inference via GSL ODE solver
   - Used as baseline/comparison for time-varying results
   - Binary version in `src/binary/` for simplified allele systems

### Key Data Flows

**HIV Workflow**:
Sequence alignment → Polymorphic site extraction → Binary trait definition (escape groups) → ODE inference → Time-varying selection coefficients

**Simulation Workflow**:
Generate population trajectories → Extract polymorphic sites → Solve forward ODEs → Infer selection → Compare to known ground-truth parameters

## Critical Conventions

### Data Input/Output Formats

- **Sequence files** (`.dat`): Tab-separated, format: `time | count | allele₀ allele₁ ... alleleₙ` (0 = ref, 1+ = mutations)
- **Escape groups**: Sites grouped by immunological epitope; stored in CSV with epitope name and polymorphic index
- **Time points**: Integer generations; multiple sequences per timepoint allowed
- **Output** (`.npz`): NumPy compressed arrays with inferred coefficients and confidence intervals

### Sequence Indexing

- **HXB2 coordinates**: HIV reference genome positions used for alignment (see `index2frame()` in epitope.py)
- **Polymorphic indexing**: 0-based site indices within alignment (gaps handled separately)
- **Codon handling**: nucleotide→amino acid translation; gaps ('-'), ambiguities ('X','?') preserved

### Parameter Patterns

Common simulation parameters (see `simulate_simple()` in simulation.py):
- `mut_rate`: per-locus mutation rate (~1e-3)
- `rec_rate`: per-locus recombination rate (~1e-3)  
- `s_ben`, `s_del`: constant selection coefficients (fitness multiplicative)
- `fi_1`, `fi_2`: time-varying selection functions (interpolated from time grid)

## Development Workflows

### Running Inference
```bash
cd src && g++ binary/main.cpp binary/inf_binary.cpp ... -O3 -std=c++11 -lgsl -lgslcblas -o mpl
cd src && ./const-sim.sh  # Batch constant-selection inference
python inference_HIV.py -tag 700010040-3 --raw  # Single time-varying inference
```

### Simulation & Analysis
```bash
python simulation.py  # Generate trajectories and infer parameters
jupyter notebook simulation_analysis.ipynb  # Analyze results
```

### Figure Generation
All manuscript figures generated in `figures.ipynb` using custom `mplot.py` utilities (publication-grade plotting)

## Key Functions by Category

**Sequence Processing** (`epitope.py`):
- `get_MSA()`, `clip_MSA()`, `filter_excess_gaps()`: Alignment management
- `codon2aa()`, `index2frame()`: Genomic coordinate translation
- `create_index()`: Extract polymorphic sites and binary traits

**Inference** (`inference_HIV.py`):
- `AnalyzeData()`: Parse patient metadata and structure data
- `getSequence()`: Convert raw sequences to state vectors and trait vectors
- ODE system solves: $\frac{dp_i}{dt} = s_i(t) \cdot p_i(1-p_i)$ (logistic selection)

**Simulation** (`simulation.py`):
- `mutation_step()`, `recombination_step()`, `offspring_step_simple()`: Wright-Fisher dynamics
- `get_fitness_simple()`: Time-varying fitness landscape
- `initial_dis()`: Population initialization with allele frequencies

## Important Implementation Details

1. **Time interpolation**: Selection coefficients are piecewise-linear functions interpolated across observation times via `scipy.interpolate.interp1d()`

2. **Escape groups**: Multiple sites can define a single binary trait; sites must have ≥2 variants at polymorphic threshold

3. **Patient tags**: Format `7DDERRXX-T` where DD=center, ER=subject, XX=repeat, T=timepoint set (3 or 5)

4. **Simulation genotypes**: Binary strings (e.g., "ATATAAT"); beneficial/deleterious/neutral sites have fixed fitness contributions

## Common Debugging

- **ODE convergence failures**: Check for extreme parameter values or too-sparse timepoints; add intermediate time steps
- **Missing escape groups**: Ensure epitope CSV has matching patient tag and site indices are correct
- **Sequence alignment gaps**: Use `filter_excess_gaps()` thresholds; gap-only sites excluded automatically
- **C++ compilation**: Requires GSL ≥2.6; on macOS with Apple Silicon: use `-mcpu=apple-a14` flag

## References to Key Patterns

- Time-varying selection setup: [inference_HIV.py#L250+](inference_HIV.py)
- Fitness landscape definition: [simulation.py#L180+](simulation.py)
- Polymorphic site extraction: [epitope.py#L613+](epitope.py)
