# Overview

## Data Structure for Generalizability Estimation

This repository implements a suite of generalizability estimators (IPSW, AIPSW, stratification, outcome modeling, calibration weighting, augmented calibration weighting).
All estimators rely on a consistent set of core variables:

### 🔷 Required Variables
#### S — Sample Membership Indicator

Binary (0/1).

S = 1 → Unit belongs to the trial / experimental sample

S = 0 → Unit belongs to the target population

This distinction is fundamental.
Generalizability problems involve transporting treatment effects from the trial (S=1) to the target population (S=0).

Important:
Outcome and treatment assignment are only observed for trial units (S=1).
Target units (S=0) contribute covariates but no outcomes and no treatment.

#### T — Treatment Assignment Inside the Trial

Binary (0/1).

T = 1 → Treated

T = 0 → Control

T is only meaningful for S=1 units. Target units (S=0) were never randomized.

Some datasets store T for all rows, but this is only to keep arrays aligned.
All estimators automatically restrict treatment information to S=1 where appropriate.

#### Y — Observed Outcome

A continuous or binary outcome.

Defined only for S=1 (trial units).

For S=0, Y is ignored (may be np.nan if desired).

All estimators automatically use:

Y[S == 1]
T[S == 1]

## How to Prepare Your Data

### 1. Your dataset must contain these three core variables
#### ✔ S — Sample indicator

Binary

1 = trial / experimental sample

0 = target population

Required for all estimators

#### ✔ T — Treatment indicator

Binary

1 = treated

0 = control

Used only for S=1 units

#### ✔ Y — Outcome

Observed only for S=1 (trial units)

Y among S=0 target units can be:

NaN

Or simply ignored

### ✅ 2. Your dataset must include a covariate matrix X

This is the set of variables that explain:

Treatment heterogeneity

Sample selection into the trial

Adjustment for generalizability

Format:

Pandas DataFrame:

df[["X1", "X2", "X3"]]

Or NumPy array:

X = np.column_stack([x1, x2, x3])

Shape must be:

(n_samples, n_features)

Used by:

IPSW / AIPSW (to estimate sampling scores)

Stratification

Outcome modeling

Calibration & augmented calibration

### ✅ 3. Your data must align row-by-row

All arrays must represent the same units in the same order:

Y[i]  → outcome for unit i (valid only if S[i] = 1)

T[i]  → treatment for unit i (valid only if S[i] = 1)

S[i]  → sample membership for unit i

X[i]  → covariates for unit i


If these arrays are misaligned (e.g., different lengths or mismatched indices), estimators will raise errors.

### ✅ 4. Recommended additional simulation variables (optional)

If you are running simulations:

✔ pS_true

True sample selection probability

Used for oracle IPSW comparison

✔ tau_true

True individual-level treatment effect

Used for benchmarking estimator accuracy

These are not required for real data, but helpful for simulation studies.

### ✅ 5. Minimal working example
Pandas DataFrame
df = pd.DataFrame({
    "Y": ...,        # outcome (observed only for S=1)
    "T": ...,        # treatment assignment (only for S=1)
    "S": ...,        # sample indicator (1=trial, 0=target)
    "X1": ...,
    "X2": ...,
})


Calling an estimator:

Y = df["Y"].values

T = df["T"].values

S = df["S"].values

X = df[["X1", "X2"]]

from methods.ipsw import estimate_ate_ipsw

result = estimate_ate_ipsw(y=Y, a=T, s=S, X=X)

print(result.ate)

### ✅ 6. Common mistakes to avoid

❌ Using T as the sample indicator
→ Must use S for trial vs target

❌ Having outcomes (Y) filled for S=0 units
→ OK in simulation, but ignored; real target units will not have Y

❌ Mismatched array lengths
→ Always check len(Y) == len(T) == len(S) == len(X)

❌ Using categorical variables in X without encoding
→ One-hot encode or convert to numeric first

❌ Using X with shape (n,)
→ Must reshape or use a DataFrame with columns

### ✅ 7. Quick sanity check

Before running any estimator, confirm:

assert df["S"].isin([0,1]).all()

assert df["T"].isin([0,1]).all()          # only matters for S=1

assert df[["X1","X2"]].shape[1] >= 1

assert len(df["Y"]) == len(df["S"]) == len(df["T"])

If you pass this checklist, all estimators in the repository will work.

