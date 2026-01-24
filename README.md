# Analysis of Turing and Floquet Instabilities in Reaction-Diffusion Systems

This repository contains MATLAB source code for analyzing spatiotemporal instabilities in reaction-diffusion systems. The code specifically focuses on:
1.  **Classical Turing Instability (TDDI):** Destabilization of uniform steady states (USS).
2.  **Floquet-Turing Instability (FTDDI):** Destabilization of limit cycle (LC) oscillations.
3.  **Averaged Jacobian Method (AJM):** An approximate method for analyzing LC stability.

## 📂 Repository Contents

### 1. Stability Space Analysis
**File:** `turing_floquet_spaces.m`
* **Purpose:** Computes the global stability map across a 2-parameter space ($a_1, a_5$).
* **Function:**
    * Identifies regions of USS vs. LC.
    * Detects TDDI for USS.
    * Detects FTDDI for LC.
* **Output:** Generates matrices and text files mapping stability codes for different regions and Turing codes across the parameter grid.

### 2. Critical Diffusion Computation (Hybrid Method)
**File:** `critical_diffusion_FTDDI_AJM.m`
* **Purpose:** Calculates the exact critical diffusion coefficient ($d_c$) required to destabilize a LC.
* **Methodology:** Uses a **Hybrid Approach**:
    1.  **Estimation:** Uses the Averaged Jacobian Method (AJM) to get an initial guess for $d_c$.
    2.  **Refinement:** Uses the exact Floquet Multiplier method with a **Secant method** and **Dynamic Thresholding** to refine the result to high precision.
* **Output:** A text file containing the parameter $a_1$, the approximate $d_c$ (AJM), and the exact $d_c$ (Floquet).

### 3. Fixed parameter FTDDI and AJM
**File:** `FTDDI_and_AJM_fixed_param.m`
* **Purpose:** Performs a detailed sweep for a fixed  parameter set to compare the FTDDI and AJM methods.
* **Function:**
    * Fixes parameters ($a_1, a_5$) corresponding to specific instability regions.
    * Sweeps through $\gamma$ and Diffusion Ratio ($d$).
    * Computes the maximum instability metric for both methods (Floquet Multiplier vs. Real Eigenvalue).
* **Output:** Detailed data files for maximum instability metrics across Gamma and Diffusion ranges.

---

## ⚙️ Usage & Dependencies

### Prerequisites
* **Parallel Computing Toolbox** (Required for `parfor` loops).
* **Optimization Toolbox** (Used for root finding/fsolve in some routines).

### Running the Code
1.  Clone the repository:
    ```bash
    git clone [https://github.com/vjuma23/Oscillatory-dynamics.git](https://github.com/vjuma23/Oscillatory-dynamics.git)
    ```
2.  Open MATLAB and navigate to the folder.
3.  Run the desired script.
    * *Note:* Scripts are configured to use `parpool` automatically. Ensure your cluster profile is active if running on a supercomputer.

---

## 👤 Author

**Victor Juma**
* Email: vjuma23@gmail.com
* *Last Updated: January 23, 2026*

## ⚖️ License
This project is licensed under the MIT License.
